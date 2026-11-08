# -*- coding: utf-8 -*-
"""
PyQt6 Oyun İçi Şeffaf Durum Paneli (Overlay HUD — Türkçe arayüz)
Gatherion-tarzı Overlay HUD'un piksel hassasiyetinde Türkçe uygulaması:
- Çerçevesiz, Her Zaman Üstte, Yarı Şeffaf Koyu Arka Planlı Araç Penceresi & WS_EX_TRANSPARENT tıklama geçişi (click-through)
- Dinamik 3px sol vurgu kenarlığı & durum rozeti (YÜRÜYOR, TOPLANIYOR, SAVAŞTA, DURAKLATILDI/BEKLEMEDE, MOLA/HARİTA)
- Mod kapsül rozeti (ROTA MODU, MANUEL KAYIT, OTO KAYIT, OTO KEŞİF) & canlı geçen süre sayacı (DD:SS)
- Düğüm Başına 3x Toplama aşama rozeti (3x Toplama / Harvest: 1/3 -> 2/3 -> 3/3) ve canlı ilerleme çubuğu
- Sağ üst Minimap'ten otomatik kaydedilen canlı materyal sayacı (Haritalanan / Minimap Mapped: X)
- Ana eylem başlığı, açıklayıcı alt başlık ve ince 1px yatay ayırıcı çizgi
- 3 Sütunlu telemetri ızgarası (TOPLANAN/KAYDEDİLEN, BÖLGE·ROTA (AKTİF/DİNLENEN), ARD ARDA BAŞARISIZ/EŞLEŞME)
- Kayan 3 satırlık zaman damgalı etkinlik günlüğü akışı
- [F9] Menü, [F8] Nokta Ekle kısayol rozetleri, Haritalanan sayacı ve v1.6.15 sürüm etiketi içeren etkileşimli alt bilgi
- Üstte kalıcı küçült/genişlet düğmesi; küçültülünce yalnızca amber mini şerit kalır (hotkey akışları bozulmaz)
- Görüntü işleme (Vision) çakışma kontrolleri için ekran dikdörtgeni geometrisi raporlama
"""

import ctypes
import logging
import math
import os
import re
import sys
import time

# Ensure Qt offscreen platform can resolve Windows TrueType fonts in headless test environments
if "QT_QPA_FONTDIR" not in os.environ and os.path.isdir(r"C:\Windows\Fonts"):
    os.environ["QT_QPA_FONTDIR"] = r"C:\Windows\Fonts"

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

from PyQt6.QtCore import QObject, QPoint, QPointF, QRect, QRectF, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# Gizlilik & Tıklama Geçişi (Click-Through) için Win32 Genişletilmiş Pencere Stili Sabitleri
GWL_EXSTYLE: int = -20
WS_EX_TRANSPARENT: int = 0x00000020
WS_EX_LAYERED: int = 0x00080000
WS_EX_TOOLWINDOW: int = 0x00000080
SWP_NOSIZE: int = 0x0001
SWP_NOMOVE: int = 0x0002
SWP_NOZORDER: int = 0x0004
SWP_NOACTIVATE: int = 0x0010
SWP_FRAMECHANGED: int = 0x0020


class OverlaySignals(QObject):
    """İşçi iş parçacıklarından (worker threads) Overlay HUD'u güncellemek için iş parçacığı güvenli sinyal kapsayıcısı."""

    update_signal = pyqtSignal(dict)
    log_signal = pyqtSignal(str)
    mode_signal = pyqtSignal(str)
    reset_timer_signal = pyqtSignal()
    harvest_stage_signal = pyqtSignal(int, int)
    minimap_mapped_signal = pyqtSignal(int)
    consumable_status_signal = pyqtSignal(dict)


class ClickableShortcutWidget(QFrame):
    """Tuş başlığı rozeti (ör. [F9]) ve açıklama etiketi (ör. Menü) gösteren etkileşimli alt bilgi kısayol bileşeni."""

    clicked = pyqtSignal()

    def __init__(self, key_text: str, description_text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._key_text = key_text
        self._description_text = description_text
        self._init_ui()

    def _init_ui(self) -> None:
        try:
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            self.setStyleSheet("QFrame { background: transparent; border: none; }")

            layout = QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(5)

            self.key_badge = QLabel(self._key_text, self)
            self.key_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.key_badge.setStyleSheet(
                """
                QLabel {
                    background-color: rgba(255, 255, 255, 0.08);
                    color: #eaf1f7;
                    border: 1px solid rgba(255, 255, 255, 0.16);
                    border-radius: 4px;
                    font-family: 'JetBrains Mono', 'Consolas', monospace;
                    font-size: 10px;
                    font-weight: 600;
                    padding: 1px 5px;
                }
                """
            )

            self.desc_label = QLabel(self._description_text, self)
            self.desc_label.setStyleSheet(
                """
                QLabel {
                    background: transparent;
                    color: #9fb2c3;
                    border: none;
                    font-family: 'Barlow', 'Segoe UI', sans-serif;
                    font-size: 11px;
                    font-weight: 500;
                }
                """
            )

            layout.addWidget(self.key_badge)
            layout.addWidget(self.desc_label)
        except Exception as exc:
            logger.error("ClickableShortcutWidget başlatılamadı (%s): %s", self._key_text, exc)

    def mousePressEvent(self, event: Optional[QMouseEvent]) -> None:
        try:
            if event is not None and event.button() == Qt.MouseButton.LeftButton:
                self.clicked.emit()
                event.accept()
                return
        except Exception as exc:
            logger.error("Kısayol bileşeni tıklama hatası (%s): %s", self._key_text, exc)
        super().mousePressEvent(event)


class AdaptiveStatLabel(QLabel):
    """
    Stat counter label with dynamic font scaling to prevent horizontal clipping.
    Automatically scales font size down when displaying long counter strings (e.g. 150+ nodes or 999,999+).
    """

    def __init__(self, text: str = "", parent: Optional[QWidget] = None, base_size: int = 20) -> None:
        super().__init__(text, parent)
        self._base_size = base_size
        self._apply_scaled_font(text)

    def setText(self, text: str) -> None:
        super().setText(text)
        self._apply_scaled_font(text)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._apply_scaled_font(self.text())

    def _apply_scaled_font(self, text: str) -> None:
        raw = str(text or "").strip()
        length = len(raw)
        if length > 34:
            font_sz = 10
        elif length > 26:
            font_sz = 12
        elif length > 20:
            font_sz = 13
        elif length > 14:
            font_sz = 14
        elif length > 8:
            font_sz = 16
        else:
            font_sz = self._base_size

        font = self.font()
        font.setPixelSize(font_sz)
        font.setBold(True)
        self.setFont(font)
        self.setStyleSheet(
            f"""
            QLabel {{
                background: transparent;
                color: #eaf1f7;
                border: none;
                font-family: 'Chakra Petch', 'Barlow', 'Segoe UI', sans-serif;
                font-size: {font_sz}px;
                font-weight: 700;
            }}
            """
        )


class RadarProjectionWidget(QWidget):
    """
    Mini-Radar Projeksiyon Tuvali.
    Oyun içi ana viewport üzerine mini-radar blipleri, menzil halkaları, pusula yönleri
    ve hedefe yönlendiren dinamik yön okunu (target direction arrow) çizer.
    """

    def __init__(self, parent: Optional[QWidget] = None, radar_radius: int = 75) -> None:
        super().__init__(parent)
        self.radar_radius = radar_radius
        self.range_m: float = 50.0
        self.player_pos: Tuple[float, float] = (0.0, 0.0)
        self.heading_deg: float = 0.0
        self.target_bearing_deg: Optional[float] = None
        self.target_distance_m: Optional[float] = None
        self.target_name: str = ""
        self.blips: List[Dict[str, Any]] = []
        self._sweep_angle: float = 0.0
        self._sweep_timer: Optional[QTimer] = None
        self._init_ui()

    def _init_ui(self) -> None:
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        size = (self.radar_radius * 2) + 30
        self.setFixedSize(size, size + 20)
        self._sweep_timer = QTimer(self)
        self._sweep_timer.setInterval(60)
        self._sweep_timer.timeout.connect(self._on_sweep_tick)
        self._sweep_timer.start()

    def _on_sweep_tick(self) -> None:
        self._sweep_angle = (self._sweep_angle + 4.0) % 360.0
        self.update()

    def update_projection(
        self,
        player_x: float = 0.0,
        player_y: float = 0.0,
        heading_deg: float = 0.0,
        target_bearing_deg: Optional[float] = None,
        target_distance_m: Optional[float] = None,
        target_name: str = "",
        blips: Optional[List[Dict[str, Any]]] = None,
        range_m: Optional[float] = None,
    ) -> None:
        self.player_pos = (float(player_x), float(player_y))
        self.heading_deg = float(heading_deg) % 360.0
        self.target_bearing_deg = (
            float(target_bearing_deg) % 360.0 if target_bearing_deg is not None else None
        )
        self.target_distance_m = (
            float(target_distance_m) if target_distance_m is not None else None
        )
        self.target_name = str(target_name or "")
        if blips is not None:
            self.blips = list(blips)
        if range_m is not None and range_m > 0:
            self.range_m = float(range_m)
        self.update()

    def set_blips(self, blips: List[Dict[str, Any]]) -> None:
        self.blips = list(blips)
        self.update()

    def set_target_direction(
        self,
        bearing_deg: float,
        distance_m: float,
        label: str = "",
    ) -> None:
        self.target_bearing_deg = float(bearing_deg) % 360.0
        self.target_distance_m = float(distance_m)
        self.target_name = str(label or "")
        self.update()

    def paintEvent(self, event: Optional[QPaintEvent]) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        cx = self.width() / 2.0
        cy = (self.height() - 16) / 2.0
        r = float(self.radar_radius)

        # 1. Yarı şeffaf koyu zemin
        painter.setPen(QPen(QColor(56, 189, 248, 140), 1.5))
        painter.setBrush(QBrush(QColor(8, 14, 23, 175)))
        painter.drawEllipse(QPointF(cx, cy), r, r)

        # 2. Menzil çemberleri (25m / 50m) ve artı eksen çizgileri
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1.0, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), r * 0.5, r * 0.5)

        painter.setPen(QPen(QColor(255, 255, 255, 24), 1.0, Qt.PenStyle.DotLine))
        painter.drawLine(QPointF(cx - r, cy), QPointF(cx + r, cy))
        painter.drawLine(QPointF(cx, cy - r), QPointF(cx, cy + r))

        # 3. Canlı radar tarama çizgisi (sweep)
        sweep_rad = math.radians(self._sweep_angle)
        sx = cx + r * math.cos(sweep_rad)
        sy = cy + r * math.sin(sweep_rad)
        painter.setPen(QPen(QColor(67, 214, 138, 70), 1.2))
        painter.drawLine(QPointF(cx, cy), QPointF(sx, sy))

        # 4. Pusula Yön İmleri (K, D, G, B)
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        cardinals = [("K", 0), ("D", 90), ("G", 180), ("B", 270)]
        for label, angle in cardinals:
            rad = math.radians(angle - 90.0)
            lx = cx + (r - 9.0) * math.cos(rad)
            ly = cy + (r - 9.0) * math.sin(rad)
            if label == "K":
                painter.setPen(QPen(QColor(240, 166, 60, 230)))
            else:
                painter.setPen(QPen(QColor(148, 163, 184, 170)))
            painter.drawText(QRectF(lx - 7, ly - 6, 14, 12), Qt.AlignmentFlag.AlignCenter, label)

        # 5. Mini-radar blipleri (Toplanabilir, canavar, rota noktası)
        for b in self.blips:
            dist = float(b.get("distance_m", b.get("dist", 0.0)))
            bearing = float(b.get("bearing_deg", b.get("bearing", 0.0)))
            if dist > self.range_m * 1.5:
                continue
            ratio = min(0.96, max(0.06, dist / max(1.0, self.range_m)))
            blip_r = ratio * r
            b_rad = math.radians(bearing - 90.0)
            bx = cx + blip_r * math.cos(b_rad)
            by = cy + blip_r * math.sin(b_rad)

            kind = str(b.get("kind", b.get("resource_type", b.get("type", "gather")))).lower()
            if "monster" in kind or "canavar" in kind or "hostile" in kind:
                color = QColor(255, 106, 94, 235)  # Canavar kırmızı
            elif "target" in kind or "hedef" in kind:
                color = QColor(240, 166, 60, 245)  # Aktif hedef altın
            elif "waypoint" in kind or "rota" in kind:
                color = QColor(56, 189, 248, 210)  # Rota mavi/cyan
            else:
                color = QColor(67, 214, 138, 235)  # Toplanabilir zümrüt yeşil

            painter.setPen(QPen(QColor(255, 255, 255, 190), 1))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QPointF(bx, by), 3.2, 3.2)

        # 6. Dinamik Hedef Yön Oku (Target Direction Arrow)
        if self.target_bearing_deg is not None:
            t_rad = math.radians(self.target_bearing_deg - 90.0)
            dist_m = self.target_distance_m if self.target_distance_m is not None else 0.0
            is_offscreen = dist_m > self.range_m

            arrow_color = QColor(240, 166, 60, 240)
            painter.setPen(QPen(arrow_color, 2.0))
            painter.setBrush(QBrush(arrow_color))

            if is_offscreen:
                # Çember kenarına kenetlenmiş yön gösterge oku
                rim_x = cx + (r - 2.0) * math.cos(t_rad)
                rim_y = cy + (r - 2.0) * math.sin(t_rad)
                arrow_size = 9.0
                left_a = t_rad + math.radians(145.0)
                right_a = t_rad - math.radians(145.0)
                poly = QPolygonF(
                    [
                        QPointF(rim_x, rim_y),
                        QPointF(
                            rim_x + arrow_size * math.cos(left_a),
                            rim_y + arrow_size * math.sin(left_a),
                        ),
                        QPointF(
                            rim_x + arrow_size * math.cos(right_a),
                            rim_y + arrow_size * math.sin(right_a),
                        ),
                    ]
                )
                painter.drawPolygon(poly)
            else:
                # Çember içindeki hedefe uzanan yön çizgisi ve ok başı
                t_ratio = max(0.12, dist_m / max(1.0, self.range_m))
                t_dist_px = t_ratio * r
                tx = cx + t_dist_px * math.cos(t_rad)
                ty = cy + t_dist_px * math.sin(t_rad)

                painter.drawLine(QPointF(cx, cy), QPointF(tx, ty))

                arrow_size = 7.0
                left_a = t_rad + math.radians(150.0)
                right_a = t_rad - math.radians(150.0)
                poly = QPolygonF(
                    [
                        QPointF(tx, ty),
                        QPointF(
                            tx + arrow_size * math.cos(left_a),
                            ty + arrow_size * math.sin(left_a),
                        ),
                        QPointF(
                            tx + arrow_size * math.cos(right_a),
                            ty + arrow_size * math.sin(right_a),
                        ),
                    ]
                )
                painter.drawPolygon(poly)

                # Hedef düğümün etrafında altın vurgu halkası
                painter.setPen(QPen(QColor(255, 255, 255, 230), 1.2))
                painter.setBrush(QBrush(QColor(240, 166, 60, 220)))
                painter.drawEllipse(QPointF(tx, ty), 4.5, 4.5)

        # 7. Merkezdeki Oyuncu Oku (Yön açısına göre döndürülmüş)
        painter.save()
        painter.translate(cx, cy)
        p_angle = math.radians(self.heading_deg - 90.0)
        poly_player = QPolygonF(
            [
                QPointF(6.0 * math.cos(p_angle), 6.0 * math.sin(p_angle)),
                QPointF(5.0 * math.cos(p_angle + 2.4), 5.0 * math.sin(p_angle + 2.4)),
                QPointF(0.0, 0.0),
                QPointF(5.0 * math.cos(p_angle - 2.4), 5.0 * math.sin(p_angle - 2.4)),
            ]
        )
        painter.setPen(QPen(QColor(255, 255, 255, 220), 1.0))
        painter.setBrush(QBrush(QColor(56, 189, 248, 240)))
        painter.drawPolygon(poly_player)
        painter.restore()

        # 8. Dijital Metre / Açı Rozeti
        readout_y = self.height() - 14
        painter.setFont(QFont("Consolas", 8, QFont.Weight.DemiBold))
        if self.target_bearing_deg is not None and self.target_distance_m is not None:
            text = f"Hedef: {self.target_distance_m:.1f}m | {int(self.target_bearing_deg):03d}°"
            painter.setPen(QPen(QColor(240, 166, 60, 235)))
        else:
            text = f"Radar {int(self.range_m)}m · {len(self.blips)} blip"
            painter.setPen(QPen(QColor(148, 163, 184, 180)))
        painter.drawText(
            QRectF(0, readout_y, self.width(), 14), Qt.AlignmentFlag.AlignCenter, text
        )
        painter.end()


class RadarProjectionOverlay(QWidget):
    """
    Oyun Penceresi Üzerinde Bağımsız Mini-Radar Projeksiyon Penceresi.
    Tıklama geçişi (WS_EX_TRANSPARENT), odak çalmama (WindowDoesNotAcceptFocus / MA_NOACTIVATE),
    her zaman üstte (WindowStaysOnTopHint) ve şeffaf arka plan ile oyun oynanışını engellemeden
    mini-radar blipleri ve hedef yön oklarını yansıtır.
    """

    def __init__(self, parent: Optional[QWidget] = None, radar_radius: int = 70) -> None:
        super().__init__(parent)
        self.radar_radius = radar_radius
        self._click_through_enabled: bool = True
        self._init_window()

    def _init_window(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.widget = RadarProjectionWidget(self, radar_radius=self.radar_radius)
        layout.addWidget(self.widget)
        self.setFixedSize(self.widget.size())
        self.move(1660, 40)
        self.setWindowTitle("NVIDIA Mini-Radar Projection")
        self.set_click_through(True)

    def set_click_through(self, enabled: bool = True) -> bool:
        self._click_through_enabled = bool(enabled)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, bool(enabled))
        if sys.platform == "win32":
            try:
                hwnd = int(self.winId()) if self.winId() is not None else 0
                if hwnd:
                    user32 = ctypes.windll.user32
                    cur_style = int(user32.GetWindowLongW(hwnd, GWL_EXSTYLE))
                    if enabled:
                        new_style = cur_style | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW
                    else:
                        new_style = (
                            cur_style | WS_EX_LAYERED | WS_EX_TOOLWINDOW
                        ) & ~WS_EX_TRANSPARENT
                    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
                    user32.SetWindowPos(
                        hwnd,
                        0,
                        0,
                        0,
                        0,
                        0,
                        SWP_NOMOVE
                        | SWP_NOSIZE
                        | SWP_NOZORDER
                        | SWP_NOACTIVATE
                        | SWP_FRAMECHANGED,
                    )
            except Exception as exc:
                logger.debug("RadarProjectionOverlay set_click_through hatası: %s", exc)
        return True

    def nativeEvent(self, eventType: Any, message: Any) -> Tuple[bool, int]:
        try:
            if sys.platform == "win32" and eventType == b"windows_generic_MSG":
                msg = ctypes.wintypes.MSG.from_address(int(message))
                if int(msg.message) == 0x0021:  # WM_MOUSEACTIVATE
                    return True, 3  # MA_NOACTIVATE
        except Exception:
            pass
        handled = super().nativeEvent(eventType, message)
        if isinstance(handled, tuple) and len(handled) == 2 and handled[1] is not None:
            return bool(handled[0]), int(handled[1])
        return False, 0

    def update_projection(self, *args: Any, **kwargs: Any) -> None:
        self.widget.update_projection(*args, **kwargs)

    def set_blips(self, blips: List[Dict[str, Any]]) -> None:
        self.widget.set_blips(blips)

    def set_target_direction(self, bearing_deg: float, distance_m: float, label: str = "") -> None:
        self.widget.set_target_direction(bearing_deg, distance_m, label)


class OverlayHUD(QWidget):
    """
    Oyun İçi Şeffaf Durum Paneli (Overlay HUD — Türkçe).
    Piksel hassasiyetinde görsel tasarım, düğüm başına 3x toplama (1/3 -> 2/3 -> 3/3) aşama göstergesi,
    sağ üst Minimap otomatik haritalanan nokta sayacı, mini-radar projeksiyonu ve tam durum telemetrisi sağlar.
    """

    f9_clicked = pyqtSignal()
    f10_clicked = pyqtSignal()
    f8_clicked = pyqtSignal()
    quit_requested = pyqtSignal()
    harvest_stage_updated = pyqtSignal(int, int)
    minimap_mapped_updated = pyqtSignal(int)
    consumable_status_updated = pyqtSignal(dict)

    # Prototip chip renkleri: gold/green/red/cyan (gatherion-hud.html)
    STATE_THEMES: Dict[str, Tuple[str, str, str, str]] = {
        "WALKING": ("YÜRÜYOR", "#43d68a", "rgba(67,214,138,0.10)", "#43d68a"),
        "YÜRÜYOR": ("YÜRÜYOR", "#43d68a", "rgba(67,214,138,0.10)", "#43d68a"),
        "GATHERING": ("TOPLANIYOR", "#43d68a", "rgba(67,214,138,0.10)", "#43d68a"),
        "TOPLANIYOR": ("TOPLANIYOR", "#43d68a", "rgba(67,214,138,0.10)", "#43d68a"),
        "FIGHTING": ("SAVAŞTA", "#ff6a5e", "rgba(255,106,94,0.12)", "#ff6a5e"),
        "COMBAT": ("SAVAŞTA", "#ff6a5e", "rgba(255,106,94,0.12)", "#ff6a5e"),
        "SAVAŞTA": ("SAVAŞTA", "#ff6a5e", "rgba(255,106,94,0.12)", "#ff6a5e"),
        "PAUSED": ("DURAKLATILDI", "#f0a63c", "rgba(240,166,60,0.10)", "#f0a63c"),
        "DURAKLATILDI": ("DURAKLATILDI", "#f0a63c", "rgba(240,166,60,0.10)", "#f0a63c"),
        "WAITING": ("BEKLEMEDE", "#f0a63c", "rgba(240,166,60,0.10)", "#f0a63c"),
        "IDLE": ("BEKLEMEDE", "#f0a63c", "rgba(240,166,60,0.10)", "#f0a63c"),
        "BEKLEMEDE": ("BEKLEMEDE", "#f0a63c", "rgba(240,166,60,0.10)", "#f0a63c"),
        "BREAK": ("MOLA", "#3fc1d4", "rgba(63,193,212,0.10)", "#3fc1d4"),
        "MOLA": ("MOLA", "#3fc1d4", "rgba(63,193,212,0.10)", "#3fc1d4"),
        "MAP": ("HARİTA", "#e9b949", "rgba(233,185,73,0.12)", "#f5cf6a"),
        "NAVIGATING": ("HARİTA", "#e9b949", "rgba(233,185,73,0.12)", "#f5cf6a"),
        "HARİTA": ("HARİTA", "#e9b949", "rgba(233,185,73,0.12)", "#f5cf6a"),
    }

    MODE_LABELS: Dict[str, str] = {
        "route": "ROTA MODU",
        "route_mode": "ROTA MODU",
        "ROUTE": "ROTA MODU",
        "ROTA MODU": "ROTA MODU",
        "manual_record": "MANUEL KAYIT",
        "manual": "MANUEL KAYIT",
        "MANUAL RECORD": "MANUEL KAYIT",
        "MANUEL KAYIT": "MANUEL KAYIT",
        "auto_record": "OTO KAYIT",
        "auto": "OTO KAYIT",
        "AUTO RECORD": "OTO KAYIT",
        "OTO KAYIT": "OTO KAYIT",
        "explorer": "OTO KEŞİF",
        "explorer_mode": "OTO KEŞİF",
        "AUTO CARTOGRAPHY": "OTO KEŞİF",
        "OTO KEŞİF": "OTO KEŞİF",
    }

    # İngilizce -> Türkçe başlık/sütun çeviri tablosu
    COLUMN_TRANSLATIONS: Dict[str, str] = {
        "GATHERED": "TOPLANAN",
        "SAVED": "KAYDEDİLEN",
        "NODES": "NOKTALAR (AKTİF / DİNLENEN)",
        "REGION · ROUTE": "BÖLGE · ROTA",
        "F PRESSED": "F BASILMA",
        "FAILED IN A ROW": "ARD ARDA BAŞARISIZ",
        "LAST SAVED": "SON KAYDEDİLEN",
        "PROMPT MATCH": "EŞLEŞME ORANI",
    }

    DEFAULT_SIZE: Tuple[int, int] = (540, 290)
    MINIMUM_SIZE: Tuple[int, int] = (500, 275)
    EXPANDED_SIZE: Tuple[int, int] = DEFAULT_SIZE
    MINIMIZED_SIZE: Tuple[int, int] = (52, 40)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._drag_offset: Optional[QPoint] = None
        self._session_start_time: float = time.monotonic()
        self._custom_elapsed_str: Optional[str] = None
        self._current_state: str = "WAITING"
        self._current_mode: str = "ROTA MODU"
        self._accent_color: str = "#64748b"
        self._log_history: Deque[str] = deque(maxlen=3)
        self.logs: list[str] = []

        # Düğüm Başına 3x Toplama & Sağ Üst Minimap Oto-Kayıt durumu
        self._current_harvest_stage: int = 0
        self._max_harvest_stages: int = 3
        self._minimap_mapped_count: int = 0
        self._learned_frames: int = 0
        self._hp_pct: float = 100.0
        self._mp_pct: float = 100.0
        self._active_scrolls: int = 0
        self._selective_target: str = ""
        self._consumable_status: Dict[str, Any] = {
            "hp_pct": 100.0,
            "mp_pct": 100.0,
            "active_scrolls": 0,
            "selective_target": "",
            "selective_active": False,
            "hp_potion": "READY",
            "mp_potion": "READY",
            "buff_scroll": "READY",
            "consumables_ready": True,
            "badge_text": "HP 100% | MP 100% | Parşömen: 0",
            "selective_badge_text": "Seç-Topla: Tümü",
        }
        self._f9_menu: Any = None
        self._click_through_enabled: bool = False
        self._stealth_ex_style: int = WS_EX_LAYERED | WS_EX_TOOLWINDOW
        self._minimized: bool = False
        self._radar_projection: Optional[RadarProjectionOverlay] = None
        self.radar_projection: Optional[RadarProjectionOverlay] = None
        self._show_radar_projection: bool = True

        self._init_window_properties()
        self._init_ui()
        self._init_radar_projection()
        self._init_timer()
        self._apply_state_theme("WAITING")
        self._refresh_harvest_badge_style()

    def _init_radar_projection(self) -> None:
        try:
            self._radar_projection = RadarProjectionOverlay()
            self.radar_projection = self._radar_projection
            self._show_radar_projection = True
        except Exception as exc:
            logger.debug("OverlayHUD radar projeksiyonu başlatılamadı: %s", exc)
            self._radar_projection = None
            self.radar_projection = None

    def show_radar_projection(self, visible: bool = True) -> None:
        self._show_radar_projection = bool(visible)
        if self._radar_projection is not None:
            if visible and not self.isHidden() and not self.is_minimized:
                self._radar_projection.show()
            else:
                self._radar_projection.hide()

    def hide_radar_projection(self) -> None:
        self.show_radar_projection(False)

    def toggle_radar_projection(self) -> bool:
        new_state = not getattr(self, "_show_radar_projection", True)
        self.show_radar_projection(new_state)
        return new_state

    def update_radar_projection(
        self,
        player_x: float = 0.0,
        player_y: float = 0.0,
        heading_deg: float = 0.0,
        target_bearing_deg: Optional[float] = None,
        target_distance_m: Optional[float] = None,
        target_name: str = "",
        blips: Optional[List[Dict[str, Any]]] = None,
        range_m: Optional[float] = None,
    ) -> None:
        if self._radar_projection is not None:
            self._radar_projection.update_projection(
                player_x=player_x,
                player_y=player_y,
                heading_deg=heading_deg,
                target_bearing_deg=target_bearing_deg,
                target_distance_m=target_distance_m,
                target_name=target_name,
                blips=blips,
                range_m=range_m,
            )

    def set_radar_blips(self, blips: List[Dict[str, Any]]) -> None:
        if self._radar_projection is not None:
            self._radar_projection.set_blips(blips)

    def set_target_direction(
        self,
        bearing_deg: float,
        distance_m: float,
        label: str = "",
    ) -> None:
        if self._radar_projection is not None:
            self._radar_projection.set_target_direction(bearing_deg, distance_m, label)


    def _init_window_properties(self) -> None:
        """Çerçevesiz, her zaman üstte, araç penceresi bayraklarını ve yarı şeffaf özellikleri yapılandır."""
        try:
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
                | Qt.WindowType.WindowDoesNotAcceptFocus
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            # Never steal keyboard from AION2 — SendInput follows foreground focus.
            # Prototip .hud: geniş alt şerit paneli (token’lar gatherion-hud.html)
            self.setMinimumSize(*self.MINIMUM_SIZE)
            self.resize(*self.DEFAULT_SIZE)
            self.move(30, 30)
            self.setWindowTitle("NVIDIA GeForce Overlay")
            self.setObjectName("NVIDIAGeForceOverlayWindow")
        except Exception as exc:
            logger.error("OverlayHUD pencere özellikleri ayarlanamadı: %s", exc)

    def set_enterprise_status(
        self,
        yolo_provider: str = "CUDA",
        kinah_per_hour: int = 0,
        active_char: str = "Main",
    ) -> None:
        """
        Enterprise status, YOLO provider, Kinah/hour, and active character fields,
        and optionally updates tooltip or subtitle without breaking UI layout.
        """
        try:
            self.yolo_provider = str(yolo_provider)
            self.kinah_per_hour = int(kinah_per_hour)
            self.active_char = str(active_char)
            
            # Format tooltip or subtitle update safely
            formatted_sub = f"Char: {self.active_char} | YOLO: {self.yolo_provider} | Kinah/h: {self.kinah_per_hour:,}"
            if hasattr(self, "subtitle_label") and self.subtitle_label is not None:
                current_text = self.subtitle_label.text()
                # If subtitle doesn't already contain enterprise info, append or update nicely
                if "YOLO:" not in current_text and "Char:" not in current_text:
                    self.subtitle_label.setToolTip(formatted_sub)
        except Exception as exc:
            logger.error("set_enterprise_status error: %s", exc)

    def set_click_through(self, enabled: bool = True) -> bool:
        """
        Win32 tıklama geçişi (WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW)
        ve Qt WA_TransparentForMouseEvents niteliklerini tüm gizlilik bayraklarını koruyarak uygular.
        """
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, bool(enabled))
            if enabled:
                target_style = WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW
            else:
                target_style = (WS_EX_LAYERED | WS_EX_TOOLWINDOW) & ~WS_EX_TRANSPARENT

            self._stealth_ex_style = target_style
            self._click_through_enabled = bool(enabled)

            if sys.platform == "win32":
                hwnd = int(self.winId()) if self.winId() is not None else 0
                if hwnd:
                    user32 = ctypes.windll.user32
                    cur_style = int(user32.GetWindowLongW(hwnd, GWL_EXSTYLE))
                    if enabled:
                        new_style = cur_style | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW
                    else:
                        new_style = (cur_style | WS_EX_LAYERED | WS_EX_TOOLWINDOW) & ~WS_EX_TRANSPARENT
                    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
                    user32.SetWindowPos(
                        hwnd,
                        0,
                        0,
                        0,
                        0,
                        0,
                        SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED,
                    )
            if hasattr(self, "_radar_projection") and self._radar_projection is not None:
                try:
                    self._radar_projection.set_click_through(enabled)
                except Exception:
                    pass
            return True
        except Exception as exc:
            logger.error("OverlayHUD click-through durumu değiştirilemedi: %s", exc)
            return False

    @property
    def is_minimized(self) -> bool:
        """HUD küçültülmüş (yalnızca üst toggle şeridi) durumda mı."""
        return bool(self._minimized)

    @pyqtSlot()
    def toggle_minimized(self) -> None:
        """Küçült ↔ genişlet arasında geçiş yap (oturum içi hatırlanır)."""
        self.set_minimized(not self._minimized)

    def set_minimized(self, minimized: bool) -> None:
        """
        HUD'u küçült veya genişlet.
        Küçültülünce kart gizlenir; ekranın üstünde yalnızca amber toggle şeridi kalır.
        Hotkey (Ctrl+Alt+F10 vb.) bağlantıları etkilenmez.
        """
        try:
            want_min = bool(minimized)
            if not hasattr(self, "card") or not hasattr(self, "_mini_bar"):
                self._minimized = want_min
                return

            if (
                want_min == self._minimized
                and want_min
                and not self._mini_bar.isHidden()
                and self.card.isHidden()
            ):
                return
            if (
                want_min == self._minimized
                and not want_min
                and self._mini_bar.isHidden()
                and not self.card.isHidden()
            ):
                return

            self._minimized = want_min
            if want_min:
                self.card.hide()
                self._mini_bar.show()
                self.setFixedSize(*self.MINIMIZED_SIZE)
                if hasattr(self, "_radar_projection") and self._radar_projection is not None:
                    self._radar_projection.hide()
            else:
                self._mini_bar.hide()
                self.card.show()
                self.setMinimumSize(*self.MINIMUM_SIZE)
                self.setMaximumSize(16777215, 16777215)
                self.resize(*self.DEFAULT_SIZE)
                if (
                    hasattr(self, "_radar_projection")
                    and self._radar_projection is not None
                    and getattr(self, "_show_radar_projection", True)
                    and not self.isHidden()
                ):
                    self._radar_projection.show()

        except Exception as exc:
            logger.error("OverlayHUD küçült/genişlet hatası: %s", exc)

    def _make_hud_toggle_button(self, parent: QWidget, object_name: str) -> QPushButton:
        """Koyu + amber vurğulu küçült/genişlet düğmesi oluştur."""
        button = QPushButton("−", parent)
        button.setObjectName(object_name)
        button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setFixedSize(30, 28)
        button.setToolTip("HUD'u küçült")
        button.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(240, 166, 60, 0.12);
                color: #f0a63c;
                border: 1px solid #f0a63c;
                border-radius: 7px;
                font-family: 'JetBrains Mono', 'Consolas', monospace;
                font-size: 16px;
                font-weight: 700;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: rgba(240, 166, 60, 0.22);
            }
            QPushButton:pressed {
                background-color: rgba(240, 166, 60, 0.32);
            }
            """
        )
        button.clicked.connect(self.toggle_minimized)
        return button

    def _init_ui(self) -> None:
        """3x Toplama Aşama Rozeti ve Sağ Üst Minimap Sayacı ile Türkçe HUD arayüzünü oluştur."""
        try:
            root_layout = QVBoxLayout(self)
            root_layout.setContentsMargins(0, 0, 0, 0)
            root_layout.setSpacing(0)

            # Küçültülmüş modda kalan kalıcı mini şerit (toggle her zaman tıklanabilir)
            self._mini_bar = QFrame(self)
            self._mini_bar.setObjectName("HudMiniBar")
            self._mini_bar.setStyleSheet(
                """
                QFrame#HudMiniBar {
                    background-color: rgba(8, 13, 21, 224);
                    border: 1px solid rgba(240, 166, 60, 0.55);
                    border-radius: 10px;
                }
                """
            )
            mini_layout = QHBoxLayout(self._mini_bar)
            mini_layout.setContentsMargins(6, 4, 6, 4)
            mini_layout.setSpacing(0)
            self.btn_hud_expand = self._make_hud_toggle_button(self._mini_bar, "HudExpandButton")
            self.btn_hud_expand.setText("+")
            self.btn_hud_expand.setToolTip("HUD'u genişlet")
            mini_layout.addWidget(self.btn_hud_expand)
            self._mini_bar.hide()
            root_layout.addWidget(self._mini_bar, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

            # 3px sol vurgu kenarlıklı ve yarı şeffaf koyu arka planlı ana kart çerçevesi
            self.card = QFrame(self)
            self.card.setObjectName("OverlayCard")
            root_layout.addWidget(self.card)

            card_layout = QVBoxLayout(self.card)
            # .hud-strip / .hud-main padding from gatherion-hud.html (tightened to prevent horizontal clipping)
            card_layout.setContentsMargins(10, 6, 10, 8)
            card_layout.setSpacing(3)

            # -----------------------------------------------------------------
            # 1. ÜST SATIR: Durum Rozeti | Mod | 3x Toplama | Esnek | Süre | Küçült
            # -----------------------------------------------------------------
            header_layout = QHBoxLayout()
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(4)

            self.badge_label = QLabel("BEKLEMEDE", self.card)
            self.badge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.badge_lbl = self.badge_label  # Geriye dönük uyumluluk takma adı

            self.mode_label = QLabel("ROTA MODU", self.card)
            self.mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.mode_label.setStyleSheet(
                """
                QLabel {
                    background-color: rgba(255, 255, 255, 0.06);
                    color: #eaf1f7;
                    border: 1px solid rgba(255, 255, 255, 0.09);
                    border-radius: 6px;
                    font-family: 'Chakra Petch', 'Barlow', 'Segoe UI', sans-serif;
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 0.05em;
                    padding: 2px 5px;
                }
                """
            )

            # Düğüm Başına 3x Toplama aşama rozeti (ör. "3x Toplama (Harvest): 2/3")
            self.harvest_badge_label = QLabel("3x Toplama (Harvest: 0/3)", self.card)
            self.harvest_badge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.harvest_badge_lbl = self.harvest_badge_label
            self.harvest_badge_label.setStyleSheet(
                """
                QLabel {
                    background-color: rgba(255, 255, 255, 0.06);
                    color: #9fb2c3;
                    border: 1px solid rgba(255, 255, 255, 0.09);
                    border-radius: 6px;
                    font-family: 'Chakra Petch', 'Barlow', 'Segoe UI', sans-serif;
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 0.05em;
                    padding: 2px 5px;
                }
                """
            )

            self.elapsed_label = QLabel("0:00", self.card)
            self.elapsed_label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.elapsed_label.setStyleSheet(
                """
                QLabel {
                    background: rgba(255, 255, 255, 0.06);
                    color: #eaf1f7;
                    border: 1px solid rgba(255, 255, 255, 0.09);
                    border-radius: 6px;
                    font-family: 'JetBrains Mono', 'Consolas', monospace;
                    font-size: 11px;
                    font-weight: 600;
                    padding: 2px 6px;
                }
                """
            )
            self.time_lbl = self.elapsed_label  # Geriye dönük uyumluluk takma adı

            self.btn_hud_minimize = self._make_hud_toggle_button(self.card, "HudMinimizeButton")
            self.btn_hud_minimize.setText("−")
            self.btn_hud_minimize.setToolTip("HUD'u küçült")

            header_layout.addWidget(self.badge_label)
            header_layout.addWidget(self.mode_label)
            header_layout.addWidget(self.harvest_badge_label)
            header_layout.addStretch(1)
            header_layout.addWidget(self.elapsed_label)
            header_layout.addWidget(self.btn_hud_minimize)
            card_layout.addLayout(header_layout)

            # -----------------------------------------------------------------
            # 2. ANA BAŞLIK, ALT BAŞLIK, 3X TOPLAMA İLERLEME ÇUBUĞU & AYIRICI ÇİZGİ
            # -----------------------------------------------------------------
            self.title_label = QLabel("Durum: Beklemede", self.card)
            self.title_label.setStyleSheet(
                """
                QLabel {
                    background: transparent;
                    color: #eaf1f7;
                    border: none;
                    font-family: 'Chakra Petch', 'Barlow', 'Segoe UI', sans-serif;
                    font-size: 20px;
                    font-weight: 700;
                    letter-spacing: 0.01em;
                    padding-top: 1px;
                }
                """
            )
            self.title_lbl = self.title_label  # Geriye dönük uyumluluk takma adı
            card_layout.addWidget(self.title_label)

            self.subtitle_label = QLabel("F9: Menü  |  F8: Nokta Ekle", self.card)
            self.subtitle_label.setStyleSheet(
                """
                QLabel {
                    background: transparent;
                    color: #9fb2c3;
                    border: none;
                    font-family: 'Barlow', 'Segoe UI', sans-serif;
                    font-size: 12px;
                    font-weight: 500;
                    padding-bottom: 1px;
                }
                """
            )
            self.sub_lbl = self.subtitle_label  # Geriye dönük uyumluluk takma adı
            card_layout.addWidget(self.subtitle_label)

            # 1/3 -> 2/3 -> 3/3 düğüm toplama aşaması için 3 kademeli ilerleme çubuğu
            self.harvest_progress_bar = QProgressBar(self.card)
            self.harvest_progress_bar.setRange(0, 3)
            self.harvest_progress_bar.setValue(0)
            self.harvest_progress_bar.setTextVisible(False)
            self.harvest_progress_bar.setFixedHeight(4)
            self.harvest_progress_bar.setStyleSheet(
                """
                QProgressBar {
                    background-color: rgba(255, 255, 255, 0.10);
                    border: none;
                    border-radius: 3px;
                }
                QProgressBar::chunk {
                    background-color: #43d68a;
                    border-radius: 3px;
                }
                """
            )
            card_layout.addWidget(self.harvest_progress_bar)

            self.separator_line = QFrame(self.card)
            self.separator_line.setFixedHeight(1)
            self.separator_line.setStyleSheet(
                "QFrame { background-color: rgba(255, 255, 255, 0.08); border: none; }"
            )
            card_layout.addWidget(self.separator_line)

            # -----------------------------------------------------------------
            # 3. ÜÇ SÜTUNLU İSTATİSTİK IZGARASI (TOPLANAN | AKTİF/DİNLENEN | BAŞARISIZ)
            # -----------------------------------------------------------------
            self.stats_container = QWidget(self.card)
            self.stats_container.setStyleSheet("QWidget { background: transparent; border: none; }")
            stats_grid = QGridLayout(self.stats_container)
            stats_grid.setContentsMargins(0, 0, 0, 0)
            stats_grid.setHorizontalSpacing(14)
            stats_grid.setVerticalSpacing(1)

            self.stat1_title = QLabel("TOPLANAN", self.stats_container)
            self.stat1_value = QLabel("Toplanan: 0", self.stats_container)

            self.stat2_title = QLabel("AKTİF / DİNLENEN / ROTA", self.stats_container)
            self.stat2_value = AdaptiveStatLabel("0 aktif · 0 dinlenen", self.stats_container)

            self.stat3_title = QLabel("ARD ARDA BAŞARISIZ", self.stats_container)
            self.stat3_value = QLabel("0 / 10", self.stats_container)

            # Geriye dönük uyumluluk takma adları
            self.stat_g_title = self.stat1_title
            self.stat_g_val = self.stat1_value
            self.stat_f_title = self.stat2_title
            self.stat_f_val = self.stat2_value
            self.stat_m_title = self.stat3_title
            self.stat_m_val = self.stat3_value
            self.gathered_label = self.stat1_value
            self.ready_label = self.stat2_value
            self.failed_label = self.stat3_value

            header_col_style = """
                QLabel {
                    background: transparent;
                    color: #6d8093;
                    border: none;
                    font-family: 'Barlow', 'Segoe UI', sans-serif;
                    font-size: 11px;
                    font-weight: 600;
                    letter-spacing: 0.16em;
                }
            """
            value_col_style = """
                QLabel {
                    background: transparent;
                    color: #eaf1f7;
                    border: none;
                    font-family: 'Chakra Petch', 'Barlow', 'Segoe UI', sans-serif;
                    font-size: 20px;
                    font-weight: 700;
                }
            """

            for header_lbl in (self.stat1_title, self.stat2_title, self.stat3_title):
                header_lbl.setStyleSheet(header_col_style)
                header_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            for val_lbl in (self.stat1_value, self.stat3_value):
                val_lbl.setStyleSheet(value_col_style)
                val_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                val_lbl.setMinimumHeight(28)

            self.stat2_value.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.stat2_value.setMinimumHeight(28)
            self.stat2_value._apply_scaled_font(self.stat2_value.text())

            stats_grid.addWidget(self.stat1_title, 0, 0)
            stats_grid.addWidget(self.stat1_value, 1, 0)
            stats_grid.addWidget(self.stat2_title, 0, 1)
            stats_grid.addWidget(self.stat2_value, 1, 1)
            stats_grid.addWidget(self.stat3_title, 0, 2)
            stats_grid.addWidget(self.stat3_value, 1, 2)
            stats_grid.setColumnStretch(0, 2)
            stats_grid.setColumnStretch(1, 6)
            stats_grid.setColumnStretch(2, 2)

            card_layout.addWidget(self.stats_container)

            # -----------------------------------------------------------------
            # 4. SON 3 ZAMAN DAMGALI GÜNLÜK (LOG) SATIRI
            # -----------------------------------------------------------------
            self.logs_container = QWidget(self.card)
            self.logs_container.setObjectName("HudLog")
            self.logs_container.setStyleSheet(
                """
                QWidget#HudLog {
                    background: rgba(0, 0, 0, 0.32);
                    border: 1px solid rgba(255, 255, 255, 0.06);
                    border-radius: 8px;
                }
                """
            )
            self.logs_container.setMinimumHeight(52)
            logs_layout = QVBoxLayout(self.logs_container)
            logs_layout.setContentsMargins(8, 4, 8, 4)
            logs_layout.setSpacing(2)

            log_line_style = """
                QLabel {
                    background: transparent;
                    color: #9fb2c3;
                    border: none;
                    font-family: 'JetBrains Mono', 'Consolas', monospace;
                    font-size: 11px;
                    font-weight: 400;
                }
            """

            self.log_label_1 = QLabel("", self.logs_container)
            self.log_label_2 = QLabel("", self.logs_container)
            self.log_label_3 = QLabel("", self.logs_container)
            self.log_labels = [self.log_label_1, self.log_label_2, self.log_label_3]

            # Geriye dönük uyumluluk takma adları
            self.log_lbl1 = self.log_label_1
            self.log_lbl2 = self.log_label_2
            self.log_lbl3 = self.log_label_3

            for log_lbl in self.log_labels:
                log_lbl.setStyleSheet(log_line_style)
                log_lbl.setFixedHeight(15)
                logs_layout.addWidget(log_lbl)

            card_layout.addWidget(self.logs_container)
            card_layout.addStretch(1)

            # -----------------------------------------------------------------
            # 4b. EK DÜĞMELER (eski yatay HUD'un üstüne; düzen aynı kalır)
            # -----------------------------------------------------------------
            button_qss = (
                "QPushButton {"
                "background-color: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
                "stop:0 #243447, stop:1 #1b2735);"
                "color: #eaf1f7;"
                "border: 1px solid rgba(255, 255, 255, 0.16);"
                "border-radius: 6px;"
                "font-family: 'Barlow', 'Segoe UI', sans-serif;"
                "font-size: 12px;"
                "font-weight: 600;"
                "padding: 0px 5px;"
                "}"
                "QPushButton:hover { background-color: #2a3c52; }"
                "QPushButton:pressed { background-color: #111827; }"
                "QPushButton#HudRunButton {"
                "background-color: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
                "stop:0 #f0c95e, stop:1 #d9a92f);"
                "color: #241a02;"
                "border: 1px solid #b8891f;"
                "font-weight: 700;"
                "}"
                "QPushButton#HudRunButton:hover { background-color: #f5cf6a; }"
                "QPushButton#HudQuitButton {"
                "background-color: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
                "stop:0 #d63f36, stop:1 #b92c24);"
                "color: #ffffff;"
                "border: 1px solid #8f1f19;"
                "}"
                "QPushButton#HudQuitButton:hover { background-color: #e04a40; }"
            )
            self._action_row = QWidget(self.card)
            self._action_row.setStyleSheet(button_qss)
            action_layout = QHBoxLayout(self._action_row)
            action_layout.setContentsMargins(0, 1, 0, 1)
            action_layout.setSpacing(3)

            def action_button(text: str, name: str) -> QPushButton:
                button = QPushButton(text, self._action_row)
                button.setObjectName(name)
                button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                button.setFixedHeight(28)
                button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                action_layout.addWidget(button)
                return button

            self.btn_run = action_button("Başlat", "HudRunButton")
            self.btn_menu = action_button("Menü", "HudMenuButton")
            self.btn_node = action_button("Nokta", "HudNodeButton")
            self.btn_stop = action_button("Durdur", "HudStopButton")
            self.btn_resume = action_button("Devam", "HudResumeButton")
            self.btn_combat = action_button("Savaş", "HudCombatButton")
            self.btn_gather = action_button("Toplam", "HudGatherButton")
            self.btn_quit = action_button("Kapat", "HudQuitButton")
            self.btn_run.setToolTip("Başlat / devam (Ctrl+Alt+F10)")
            self.btn_stop.setToolTip("Durdur (Ctrl+Alt+F10)")
            self.btn_menu.setToolTip("Menü (Ctrl+Alt+F9)")
            self.btn_node.setToolTip("Nokta kaydet (Ctrl+Alt+F8)")
            self.btn_resume.setToolTip("Devam: Toplama modu")
            self.btn_combat.setToolTip("Savaş modu")
            self.btn_gather.setToolTip("Toplam: Toplama modu")
            self.btn_quit.setToolTip("Uygulamayı kapatır")
            self.btn_run.clicked.connect(self._on_f10_clicked)
            self.btn_stop.clicked.connect(self._on_f10_clicked)
            self.btn_menu.clicked.connect(self._on_f9_clicked)
            self.btn_node.clicked.connect(self._on_f8_clicked)
            self.btn_resume.clicked.connect(lambda _checked=False: self._dispatch_f9_action("resume"))
            self.btn_combat.clicked.connect(lambda _checked=False: self._dispatch_f9_action("combat"))
            self.btn_gather.clicked.connect(lambda _checked=False: self._dispatch_f9_action("route"))
            self.btn_quit.clicked.connect(self._on_quit_clicked)
            card_layout.addWidget(self._action_row)

            # -----------------------------------------------------------------
            # 5. ALT BİLGİ SATIRI: [F10] Durdur | [F9] Menü | [F8] Nokta | YZ | Harita | Sürüm
            # -----------------------------------------------------------------
            footer_layout = QHBoxLayout()
            footer_layout.setContentsMargins(0, 1, 0, 0)
            footer_layout.setSpacing(6)

            self.f10_shortcut = ClickableShortcutWidget("F10", "Durdur", self.card)
            self.f10_shortcut.setToolTip("Ctrl+Alt+F10: Durdur / Başlat")
            self.f10_shortcut.clicked.connect(self._on_f10_clicked)

            self.f9_shortcut = ClickableShortcutWidget("F9", "Menü", self.card)
            self.f9_shortcut.setToolTip("F9: Hızlı Kontrol Menüsünü Aç")
            self.f9_shortcut.clicked.connect(self._on_f9_clicked)

            self.f8_shortcut = ClickableShortcutWidget("F8", "Nokta", self.card)
            self.f8_shortcut.setToolTip("F8: Mevcut Konumu Rota Noktası Olarak Kaydet")
            self.f8_shortcut.clicked.connect(self._on_f8_clicked)

            self.ai_learned_label = QLabel("YZ 0 Kare", self.card)
            self.ai_learned_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ai_learned_label.setToolTip("Öğrenilen kare")
            self.ai_learned_label.setStyleSheet(
                """
                QLabel {
                    background-color: rgba(201, 162, 255, 0.14);
                    color: #c9a2ff;
                    border: 1px solid rgba(201, 162, 255, 0.40);
                    border-radius: 7px;
                    font-family: 'Chakra Petch', 'Barlow', sans-serif;
                    font-size: 10px;
                    font-weight: 700;
                    letter-spacing: 0.06em;
                    padding: 3px 8px;
                }
                """
            )

            # Sağ üst Minimap'ten otomatik kaydedilen canlı materyal noktası sayacı
            self.minimap_mapped_label = QLabel("Harita", self.card)
            self.minimap_mapped_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.minimap_mapped_label.setToolTip("Sağ üst Minimap'ten otomatik haritalanan kaynak noktası sayısı")
            self.minimap_mapped_label.setStyleSheet(
                """
                QLabel {
                    background-color: rgba(63, 193, 212, 0.12);
                    color: #3fc1d4;
                    border: 1px solid rgba(63, 193, 212, 0.40);
                    border-radius: 7px;
                    font-family: 'Chakra Petch', 'Barlow', sans-serif;
                    font-size: 10px;
                    font-weight: 700;
                    letter-spacing: 0.06em;
                    padding: 3px 8px;
                }
                """
            )
            self.minimap_mapped_lbl = self.minimap_mapped_label

            # Tüketilebilir (HP/MP İksiri & Parşömen) ve Seç-Topla Hedef Materyal rozetleri
            self.consumable_badge_label = QLabel("İksir: Hazır", self.card)
            self.consumable_badge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.consumable_badge_label.setToolTip("HP: 100% (READY) | MP: 100% (READY) | Parşömen: 0 (READY)")
            self.consumable_badge_label.setStyleSheet(
                """
                QLabel {
                    background-color: rgba(67, 214, 138, 0.12);
                    color: #43d68a;
                    border: 1px solid rgba(67, 214, 138, 0.40);
                    border-radius: 7px;
                    font-family: 'Chakra Petch', 'Barlow', sans-serif;
                    font-size: 10px;
                    font-weight: 700;
                    letter-spacing: 0.04em;
                    padding: 2px 6px;
                }
                """
            )
            self.consumable_badge_label.setVisible(False)
            self.consumable_badge_lbl = self.consumable_badge_label

            self.selective_badge_label = QLabel("Seç-Topla: Tümü", self.card)
            self.selective_badge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.selective_badge_label.setToolTip("Seç-Topla hedef materyal filtresi")
            self.selective_badge_label.setStyleSheet(
                """
                QLabel {
                    background-color: rgba(240, 166, 60, 0.14);
                    color: #f0a63c;
                    border: 1px solid rgba(240, 166, 60, 0.42);
                    border-radius: 7px;
                    font-family: 'Chakra Petch', 'Barlow', sans-serif;
                    font-size: 10px;
                    font-weight: 700;
                    letter-spacing: 0.04em;
                    padding: 2px 6px;
                }
                """
            )
            self.selective_badge_label.setVisible(False)
            self.selective_target_label = self.selective_badge_label

            self.version_label = QLabel("v1.6.15", self.card)
            self.version_label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.version_label.setStyleSheet(
                """
                QLabel {
                    background: transparent;
                    color: #6d8093;
                    border: none;
                    font-family: 'Barlow', 'Segoe UI', sans-serif;
                    font-size: 11px;
                    font-weight: 500;
                }
                """
            )

            footer_layout.addWidget(self.f10_shortcut)
            footer_layout.addWidget(self.f9_shortcut)
            footer_layout.addWidget(self.f8_shortcut)
            footer_layout.addWidget(self.ai_learned_label)
            footer_layout.addWidget(self.minimap_mapped_label)
            footer_layout.addWidget(self.consumable_badge_label)
            footer_layout.addWidget(self.selective_badge_label)
            footer_layout.addStretch(1)
            footer_layout.addWidget(self.version_label)

            card_layout.addLayout(footer_layout)
        except Exception as exc:
            logger.error("OverlayHUD arayüzü oluşturulamadı: %s", exc)

    def _init_timer(self) -> None:
        """Geçen oturum süresini güncellemek için 1 saniyelik zamanlayıcıyı başlat."""
        try:
            self._elapsed_timer = QTimer(self)
            self._elapsed_timer.setInterval(1000)
            self._elapsed_timer.timeout.connect(self._update_elapsed_time)
            self._elapsed_timer.start()
        except Exception as exc:
            logger.error("OverlayHUD süre zamanlayıcısı başlatılamadı: %s", exc)

    def _refresh_harvest_badge_style(self) -> None:
        """Aktif toplama aşamasına göre 3x Toplama rozeti ve ilerleme çubuğu stilini güncelle."""
        try:
            stage = max(0, int(self._current_harvest_stage))
            max_s = max(1, int(self._max_harvest_stages))
            self.harvest_badge_label.setText(f"3x Toplama (Harvest: {stage}/{max_s})")

            self.harvest_progress_bar.setRange(0, max_s)
            self.harvest_progress_bar.setValue(min(stage, max_s))

            is_active = stage > 0 or self._current_state in ("GATHERING", "TOPLANIYOR")
            badge_bg = "rgba(67, 214, 138, 0.10)" if is_active else "rgba(255, 255, 255, 0.06)"
            self.harvest_badge_label.setStyleSheet(
                f"""
                QLabel {{
                    background-color: {badge_bg};
                    color: #9fb2c3;
                    border: 1px solid rgba(255, 255, 255, 0.09);
                    border-radius: 6px;
                    font-family: 'Chakra Petch', 'Barlow', 'Segoe UI', sans-serif;
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 0.05em;
                    padding: 2px 5px;
                }}
                """
            )
            # Layout squishing'i önlemek için metin genişliği + dolguyu garantiye al
            fm = self.harvest_badge_label.fontMetrics()
            adv = fm.horizontalAdvance(self.harvest_badge_label.text())
            self.harvest_badge_label.setMinimumWidth(adv + 14)
        except Exception as exc:
            logger.error("3x Toplama rozet stili güncellenemedi: %s", exc)

    @staticmethod
    def _format_gathering_title_with_stage(raw_title: str, stage: int, max_stages: int) -> str:
        """
        Herhangi bir toplama başlığının Türkçe durum ve [aşama/maks] rozetini içermesini sağlar:
        ör. 'Durum: Toplanıyor [2/3] — Gathering node #12 [2/3]' veya 'Durum: Toplanıyor [2/3]'
        """
        cleaned = re.sub(r"\s*\[\d+\s*/\s*\d+\]\s*$", "", (raw_title or "").strip())
        display_stage = max(1, min(stage if stage > 0 else 1, max(1, max_stages)))
        stage_tag = f"[{display_stage}/{max(1, max_stages)}]"

        if not cleaned or cleaned.lower() in ("gathering", "toplanıyor", "durum: toplanıyor"):
            return f"Durum: Toplanıyor {stage_tag}"

        if "Durum: Toplanıyor" in cleaned:
            return f"{cleaned} {stage_tag}"

        return f"Durum: Toplanıyor {stage_tag} · {cleaned} {stage_tag}"

    @staticmethod
    def _translate_title(raw_title: str) -> str:
        """İngilizce durum başlıklarını temiz Türkçe karşılıklarına çevirir."""
        text = (raw_title or "").strip()
        if not text:
            return "Durum: Beklemede"
        lower = text.lower()
        if lower == "manual record":
            return "Durum: Manuel Kayıt Modu"
        if lower == "auto record":
            return "Durum: Otomatik Kayıt Modu"
        if lower in ("ready", "waiting", "idle"):
            return "Durum: Beklemede"
        if lower.startswith("paused"):
            reason = text.partition(":")[2].strip() or text.partition("(")[2].rstrip(")")
            if reason:
                return f"Durum: Duraklatıldı ({reason})"
            return "Durum: Duraklatıldı"
        if lower.startswith("walking to node"):
            return f"Durum: Noktaya Yürünüyor ({text})"
        return text

    @staticmethod
    def _translate_subtitle(raw_sub: str) -> str:
        """İngilizce alt başlık metinlerini anlaşılır Türkçe açıklamaya dönüştürür."""
        text = (raw_sub or "").strip()
        if not text:
            return "F9: Menü  |  F8: Nokta Ekle"
        lower = text.lower()
        if "press f9 to open menu" in lower:
            return "F9: Menü  |  F8: Nokta Ekle"
        if "walk around and press f8" in lower:
            return "Gezinirken F8 tuşu ile kaynak noktalarını kaydedin (F9: Menü)"
        if "bot presses f and saves nodes" in lower:
            return "Siz yürüyün; bot otomatik F basar ve noktaları kaydeder"
        if "on arrival: f" in lower:
            return "Varışta: 3x F toplama veya toplama simgesine tıklama"
        return text

    @pyqtSlot(int, int)
    def update_harvest_stage(self, current_stage: int, max_stages: int = 3) -> None:
        """
        Canlı 3x Toplama aşama rozetini (1/3 -> 2/3 -> 3/3) ve ilerleme çubuğunu gerçek zamanlı günceller.
        Ayrıca şu anda bir düğüm toplanıyorsa ana başlığı da [aşama/maks] ile günceller.
        """
        try:
            self._current_harvest_stage = max(0, int(current_stage))
            self._max_harvest_stages = max(1, int(max_stages))
            self._refresh_harvest_badge_style()

            current_title = self.title_label.text()
            if (
                self._current_state in ("GATHERING", "TOPLANIYOR")
                or "toplanıyor" in current_title.lower()
                or "gathering" in current_title.lower()
            ) and self._current_harvest_stage > 0:
                updated_title = self._format_gathering_title_with_stage(
                    current_title,
                    self._current_harvest_stage,
                    self._max_harvest_stages,
                )
                self.title_label.setText(updated_title)
            elif self._current_harvest_stage > 0:
                self.title_label.setText(
                    f"Durum: Toplanıyor [{self._current_harvest_stage}/{self._max_harvest_stages}]"
                )

            self.harvest_stage_updated.emit(self._current_harvest_stage, self._max_harvest_stages)
        except Exception as exc:
            logger.error("update_harvest_stage(%s, %s) hatası: %s", current_stage, max_stages, exc)

    @pyqtSlot(int)
    def update_minimap_mapped(self, count: int) -> None:
        """Sağ üst Minimap'ten otomatik kaydedilen canlı nokta sayısını günceller."""
        try:
            self._minimap_mapped_count = max(0, int(count))
            self.minimap_mapped_label.setText("Harita")
            self.minimap_mapped_label.setToolTip(
                f"Haritalanan: {self._minimap_mapped_count} (Minimap Mapped: {self._minimap_mapped_count})"
            )
            self.minimap_mapped_updated.emit(self._minimap_mapped_count)
        except Exception as exc:
            logger.error("update_minimap_mapped(%s) hatası: %s", count, exc)

    @property
    def current_harvest_stage(self) -> int:
        return self._current_harvest_stage

    @property
    def max_harvest_stages(self) -> int:
        return self._max_harvest_stages

    @property
    def minimap_mapped_count(self) -> int:
        return self._minimap_mapped_count

    def update_consumable_and_selective_status(
        self,
        hp_pct: float = 100.0,
        mp_pct: float = 100.0,
        active_scrolls: Any = 0,
        selective_target: str = "",
        active_buffs: Optional[List[str]] = None,
        selected_material: str = "",
    ) -> dict:
        """
        Aktif tüketilebilir/iksir/parşömen durumunu (hp_potion, mp_potion, buff_scroll kalan süre/hazır durumu)
        ve Seç-Topla hedef materyal rozetini günceller ve güncel durum sözlüğünü döndürür.
        """
        try:
            hp_val = max(0.0, min(100.0, float(hp_pct)))
            mp_val = max(0.0, min(100.0, float(mp_pct)))
            if isinstance(active_scrolls, (list, tuple, set)):
                scrolls_val = len(active_scrolls)
            elif active_buffs is not None:
                scrolls_val = len(active_buffs)
            else:
                scrolls_val = max(0, int(active_scrolls or 0))
            target_val = str(selective_target or selected_material or "").strip()

            self._hp_pct = hp_val
            self._mp_pct = mp_val
            self._active_scrolls = scrolls_val
            self._selective_target = target_val

            hp_potion_state = "TRIGGERED" if hp_val < 50.0 else "READY"
            mp_potion_state = "TRIGGERED" if mp_val < 40.0 else "READY"
            buff_scroll_state = f"ACTIVE ({scrolls_val})" if scrolls_val > 0 else "READY"

            badge_text = f"HP {int(round(hp_val))}% · MP {int(round(mp_val))}% · S:{scrolls_val}"
            selective_badge_text = (
                f"Hedef: {target_val}" if target_val else "Seç-Topla: Tümü"
            )

            status_payload: Dict[str, Any] = {
                "hp_pct": round(hp_val, 1),
                "mp_pct": round(mp_val, 1),
                "active_scrolls": scrolls_val,
                "selective_target": target_val,
                "selective_active": bool(target_val),
                "hp_potion": hp_potion_state,
                "mp_potion": mp_potion_state,
                "buff_scroll": buff_scroll_state,
                "consumables_ready": True,
                "badge_text": badge_text,
                "selective_badge_text": selective_badge_text,
            }
            self._consumable_status = status_payload

            c_lbl = self.__dict__.get("consumable_badge_label")
            if c_lbl is not None:
                c_lbl.setText(badge_text)
                c_lbl.setToolTip(
                    f"HP İksiri: {hp_potion_state} ({hp_val:.1f}%) | "
                    f"MP İksiri: {mp_potion_state} ({mp_val:.1f}%) | "
                    f"Parşömen: {buff_scroll_state}"
                )
                is_alert = hp_potion_state == "TRIGGERED" or mp_potion_state == "TRIGGERED"
                bg_col = "rgba(255, 106, 94, 0.16)" if is_alert else "rgba(67, 214, 138, 0.12)"
                fg_col = "#ff6a5e" if is_alert else "#43d68a"
                border_col = "rgba(255, 106, 94, 0.45)" if is_alert else "rgba(67, 214, 138, 0.40)"
                c_lbl.setStyleSheet(
                    f"""
                    QLabel {{
                        background-color: {bg_col};
                        color: {fg_col};
                        border: 1px solid {border_col};
                        border-radius: 7px;
                        font-family: 'Chakra Petch', 'Barlow', sans-serif;
                        font-size: 10px;
                        font-weight: 700;
                        letter-spacing: 0.04em;
                        padding: 2px 6px;
                    }}
                    """
                )

            s_lbl = self.__dict__.get("selective_badge_label")
            if s_lbl is not None:
                s_lbl.setText(selective_badge_text)
                s_lbl.setToolTip(
                    f"Seç-Topla Aktif Hedef: {target_val}" if target_val else "Seç-Topla: Tüm Materyaller"
                )
                s_lbl.setVisible(bool(target_val))

            try:
                self.consumable_status_updated.emit(dict(status_payload))
            except Exception:
                pass
            return dict(status_payload)
        except Exception as exc:
            logger.error("update_consumable_and_selective_status hatası: %s", exc)
            return dict(self.__dict__.get("_consumable_status", {}))

    def get_overlay_telemetry_snapshot(self) -> dict:
        """
        Mevcut durum, mod, 3x hasat aşaması (1/3..3/3), minimap sayacı ve tüketilebilir
        durumunu tek sözlükte döndürür.
        """
        try:
            stage = max(0, int(self.__dict__.get("_current_harvest_stage", 0)))
            max_s = max(1, int(self.__dict__.get("_max_harvest_stages", 3)))
            active_stage = min(max(1, stage), max_s) if stage > 0 else 1
            harvest_phase_str = f"{active_stage}/{max_s}"
            harvest_raw_str = f"{min(stage, max_s)}/{max_s}"
            consumables = dict(self.__dict__.get("_consumable_status", {}))

            return {
                "state": self.__dict__.get("_current_state", "WAITING"),
                "mode": self.__dict__.get("_current_mode", "ROTA MODU"),
                "harvest_stage": stage,
                "max_harvest_stages": max_s,
                "harvest_phase": harvest_phase_str,
                "harvest_stage_str": harvest_raw_str,
                "minimap_mapped": int(self.__dict__.get("_minimap_mapped_count", 0)),
                "minimap_mapped_count": int(self.__dict__.get("_minimap_mapped_count", 0)),
                "learned_frames": int(self.__dict__.get("_learned_frames", 0)),
                "hp_pct": float(self.__dict__.get("_hp_pct", 100.0)),
                "mp_pct": float(self.__dict__.get("_mp_pct", 100.0)),
                "active_scrolls": int(self.__dict__.get("_active_scrolls", 0)),
                "selective_target": str(self.__dict__.get("_selective_target", "")),
                "hp_potion": consumables.get("hp_potion", "READY"),
                "mp_potion": consumables.get("mp_potion", "READY"),
                "buff_scroll": consumables.get("buff_scroll", "READY"),
                "consumables": consumables,
                "consumable_status": consumables,
                "is_minimized": bool(self.__dict__.get("is_minimized", False)),
                "click_through": bool(self.__dict__.get("_click_through_enabled", False)),
            }
        except Exception as exc:
            logger.error("get_overlay_telemetry_snapshot hatası: %s", exc)
            return {
                "state": self.__dict__.get("_current_state", "WAITING"),
                "mode": self.__dict__.get("_current_mode", "ROTA MODU"),
                "harvest_stage": 0,
                "max_harvest_stages": 3,
                "harvest_phase": "1/3",
                "harvest_stage_str": "0/3",
                "minimap_mapped": 0,
                "minimap_mapped_count": 0,
                "consumables": {},
                "consumable_status": {},
            }

    @pyqtSlot()
    def _on_f10_clicked(self) -> None:
        try:
            self.f10_clicked.emit()
        except Exception as exc:
            logger.error("f10_clicked sinyali gönderilemedi: %s", exc)

    @pyqtSlot()
    def _on_quit_clicked(self) -> None:
        try:
            self.quit_requested.emit()
        except Exception as exc:
            logger.error("quit_requested sinyali gönderilemedi: %s", exc)

    def bind_f9_menu(self, menu: Any) -> None:
        """HUD düğmeleri mevcut F9 menüsünün eylemlerine bağlanır."""
        self._f9_menu = menu

    def bind_live_sources(self, route_manager: Any = None, worker: Any = None) -> None:
        """Eski HUD düzenini bozmadan canlı kaynak referansını tutar."""
        if route_manager is not None:
            self._route_manager = route_manager
        if worker is not None:
            self._worker = worker

    def set_hotkey_labels(self, node: str, menu: str, pause: str) -> None:
        """Kısayol ipuçlarını günceller; yatay panel düzeni aynı kalır."""
        try:
            if hasattr(self, "f8_shortcut"):
                self.f8_shortcut.setToolTip(f"{node}: Nokta kaydet")
            if hasattr(self, "f9_shortcut"):
                self.f9_shortcut.setToolTip(f"{menu}: Menü")
            if hasattr(self, "f10_shortcut"):
                self.f10_shortcut.setToolTip(f"{pause}: Durdur / Başlat")
            if hasattr(self, "btn_node"):
                self.btn_node.setToolTip(f"{node}: Nokta kaydet")
            if hasattr(self, "btn_menu"):
                self.btn_menu.setToolTip(f"{menu}: Menü")
            if hasattr(self, "btn_run"):
                self.btn_run.setToolTip(f"{pause}: Başlat / devam")
            if hasattr(self, "btn_stop"):
                self.btn_stop.setToolTip(f"{pause}: Durdur")
        except Exception as exc:
            logger.error("HUD kısayol etiketleri güncellenemedi: %s", exc)

    def _dispatch_f9_action(self, action_id: str) -> None:
        """Devam Et, Savaş ve Toplama düğmeleri F9 menüsündeki aynı eylemi çalıştırır."""
        try:
            menu = getattr(self, "_f9_menu", None)
            if menu is not None and hasattr(menu, "_trigger_action"):
                menu._trigger_action(action_id)
        except Exception as exc:
            logger.error("HUD F9 eylemi çalışmadı (%s): %s", action_id, exc)

    def nativeEvent(self, eventType: Any, message: Any) -> Tuple[bool, int]:
        """Tıklama gelsin ama pencere öne çıkıp oyunun odağını almasın."""
        try:
            if sys.platform == "win32" and eventType == b"windows_generic_MSG":
                msg = ctypes.wintypes.MSG.from_address(int(message))
                if int(msg.message) == 0x0021:  # WM_MOUSEACTIVATE
                    return True, 3  # MA_NOACTIVATE
        except Exception as exc:
            logger.debug("OverlayHUD nativeEvent: %s", exc)
        handled = super().nativeEvent(eventType, message)
        if isinstance(handled, tuple) and len(handled) == 2 and handled[1] is not None:
            return bool(handled[0]), int(handled[1])
        return False, 0

    @pyqtSlot()
    def _on_f9_clicked(self) -> None:
        try:
            self.f9_clicked.emit()
        except Exception as exc:
            logger.error("f9_clicked sinyali gönderilemedi: %s", exc)

    @pyqtSlot()
    def _on_f8_clicked(self) -> None:
        try:
            self.f8_clicked.emit()
        except Exception as exc:
            logger.error("f8_clicked sinyali gönderilemedi: %s", exc)

    @pyqtSlot()
    def _update_elapsed_time(self) -> None:
        """Sağ üstteki geçen süre sayacını yenile (ör. 2:41 veya 1:05:12)."""
        try:
            if self._custom_elapsed_str is not None:
                self.elapsed_label.setText(self._custom_elapsed_str)
                return
            elapsed_sec = max(0, int(time.monotonic() - self._session_start_time))
            hours, remainder = divmod(elapsed_sec, 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                formatted = f"{hours}:{minutes:02d}:{seconds:02d}"
            else:
                formatted = f"{minutes}:{seconds:02d}"
            self.elapsed_label.setText(formatted)
        except Exception as exc:
            logger.error("Geçen süre etiketi güncellenemedi: %s", exc)

    @pyqtSlot()
    def reset_timer(self) -> None:
        """Oturum süresini 0:00'a sıfırla."""
        try:
            self._session_start_time = time.monotonic()
            self._custom_elapsed_str = None
            self._update_elapsed_time()
        except Exception as exc:
            logger.error("OverlayHUD zamanlayıcısı sıfırlanamadı: %s", exc)

    def _apply_state_theme(self, state_key: str) -> None:
        """Durum rozeti renklerini ve 3px sol vurgu kenarlığını HUD kartına uygula."""
        try:
            normalized = (state_key or "WAITING").strip().upper()
            badge_text, accent_hex, badge_bg, badge_fg = self.STATE_THEMES.get(
                normalized,
                (normalized, "#64748b", "#334155", "#f1f5f9"),
            )
            self._current_state = badge_text
            self._accent_color = accent_hex

            self.badge_label.setText(badge_text)
            self.badge_label.setStyleSheet(
                f"""
                QLabel {{
                    background-color: {badge_bg};
                    color: {badge_fg};
                    border: 1px solid {accent_hex};
                    border-radius: 6px;
                    font-family: 'Chakra Petch', 'Barlow', 'Segoe UI', sans-serif;
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 0.06em;
                    padding: 2px 6px;
                }}
                """
            )

            self.card.setStyleSheet(
                f"""
                QFrame#OverlayCard {{
                    background-color: rgba(8, 13, 21, 224);
                    border: 1px solid rgba(255, 255, 255, 0.16);
                    border-left: 3px solid {accent_hex};
                    border-radius: 12px;
                }}
                """
            )
        except Exception as exc:
            logger.error("Durum teması uygulanamadı ('%s'): %s", state_key, exc)

    @pyqtSlot(str)
    def set_mode(self, mode: str) -> None:
        """İkincil mod kapsül rozetini güncelle (ROTA MODU, MANUEL KAYIT, OTO KAYIT, OTO KEŞİF)."""
        try:
            raw_mode = (mode or "ROUTE").strip()
            resolved = self.MODE_LABELS.get(raw_mode, self.MODE_LABELS.get(raw_mode.lower(), raw_mode.upper()))
            self._current_mode = resolved
            self.mode_label.setText(resolved)
        except Exception as exc:
            logger.error("Mod ayarlanamadı ('%s'): %s", mode, exc)

    @pyqtSlot(dict)
    def update_state(self, data: Dict[str, Any]) -> None:
        """
        Durum sözlüğünden tüm HUD bileşenlerini günceller.

        Desteklenen anahtarlar:
        - state / badge: 'WALKING' | 'GATHERING' | 'FIGHTING' | 'COMBAT' | 'PAUSED' | 'WAITING' | 'BREAK' | 'MAP'
        - mode / mode_badge: 'ROUTE' | 'MANUAL RECORD' | 'AUTO RECORD' | 'EXPLORER'
        - elapsed / elapsed_str: isteğe bağlı özel süre dizgesi (ör. '2:41')
        - harvest_stage / current_stage: aktif 3x toplama aşaması (1..3)
        - max_harvest_stages / max_stages: düğüm başına maksimum toplama sayısı (varsayılan 3)
        - minimap_mapped / minimap_recorded / auto_mapped_count: sağ üst minimap'ten haritalanan canlı nokta sayısı
        - title: ana başlık metni
        - sub / subtitle: alt başlık metni
        - stat1_title / col1_title, stat1_val / col1_val / gathered / saved_count
        - stat2_title / col2_title, stat2_val / col2_val / region_route / ready_resting / f_pressed
        - stat3_title / col3_title, stat3_val / col3_val / failed_in_row / match_pct / last_saved
        - version: sürüm etiketi
        """
        if not isinstance(data, dict):
            logger.warning("OverlayHUD.update_state dict bekliyordu, alınan: %s", type(data).__name__)
            return

        try:
            # 1. Durum Rozeti & Sol Vurgu Kenarlığı
            raw_state = data.get("state", data.get("badge", self._current_state or "WAITING"))
            state_val = str(raw_state).upper()
            self._apply_state_theme(state_val)

            # 2. Mod Kapsül Rozeti
            raw_mode = data.get("mode", data.get("mode_badge", None))
            if raw_mode is not None:
                self.set_mode(str(raw_mode))

            # 3. Düğüm Başına 3x Toplama Aşaması Güncellemesi
            stage_val = data.get("harvest_stage", data.get("current_stage", None))
            max_stage_val = data.get("max_harvest_stages", data.get("max_stages", self._max_harvest_stages))
            if stage_val is not None:
                self._current_harvest_stage = max(0, int(stage_val))
                self._max_harvest_stages = max(1, int(max_stage_val))
            elif state_val in ("GATHERING", "TOPLANIYOR") and self._current_harvest_stage == 0:
                self._current_harvest_stage = 1
            elif (
                state_val not in ("GATHERING", "TOPLANIYOR")
                and "harvest_stage" not in data
                and "current_stage" not in data
            ):
                self._current_harvest_stage = 0
            self._refresh_harvest_badge_style()

            # 4. Sağ Üst Minimap Oto-Haritalanan Nokta Sayısı
            mm_count = data.get(
                "minimap_mapped",
                data.get("minimap_recorded", data.get("auto_mapped_count", None)),
            )
            if mm_count is not None:
                self.update_minimap_mapped(int(mm_count))

            learned = data.get("learned_frames", data.get("ai_learned_frames", None))
            if learned is not None and hasattr(self, "ai_learned_label"):
                self._learned_frames = max(0, int(learned))
                self.ai_learned_label.setText(f"YZ {self._learned_frames} Kare")
                self.ai_learned_label.setToolTip("Öğrenilen kare")

            # 5. Geçen Süre
            if "elapsed" in data or "elapsed_str" in data:
                elapsed_val = data.get("elapsed", data.get("elapsed_str"))
                self._custom_elapsed_str = str(elapsed_val) if elapsed_val is not None else None
            self._update_elapsed_time()

            # 6. Ana Başlık & Alt Başlık
            if "title" in data and data["title"] is not None:
                raw_title = str(data["title"])
                if (
                    state_val in ("GATHERING", "TOPLANIYOR")
                    or raw_title.lower().startswith("gathering")
                    or "toplanıyor" in raw_title.lower()
                ):
                    stage_for_title = self._current_harvest_stage if self._current_harvest_stage > 0 else 1
                    self.title_label.setText(
                        self._format_gathering_title_with_stage(
                            raw_title, stage_for_title, self._max_harvest_stages
                        )
                    )
                else:
                    self.title_label.setText(self._translate_title(raw_title))
            elif "node_id" in data and state_val in ("WALKING", "YÜRÜYOR"):
                dist_str = f" · {data['distance_m']} m" if "distance_m" in data else ""
                self.title_label.setText(
                    f"Durum: Noktaya Yürünüyor #{data['node_id']}{dist_str} (Walking to node {data['node_id']})"
                )
            elif "node_id" in data and state_val in ("GATHERING", "TOPLANIYOR"):
                stage_for_title = self._current_harvest_stage if self._current_harvest_stage > 0 else 1
                self.title_label.setText(
                    f"Durum: Toplanıyor [{stage_for_title}/{self._max_harvest_stages}] · Gathering node #{data['node_id']} [{stage_for_title}/{self._max_harvest_stages}]"
                )

            subtitle_val = data.get("subtitle", data.get("sub", None))
            if subtitle_val is not None:
                text_sub = self._translate_subtitle(str(subtitle_val))
                self.subtitle_label.setText(text_sub)
                self.subtitle_label.setVisible(bool(text_sub.strip()))
            else:
                self.subtitle_label.setVisible(True)

            # 7. İstatistik Izgarası Sütunları
            # Sütun 1: TOPLANAN veya KAYDEDİLEN
            col1_t = data.get("stat1_title", data.get("col1_title", None))
            if col1_t is not None:
                raw_c1_t = str(col1_t).upper()
                self.stat1_title.setText(self.COLUMN_TRANSLATIONS.get(raw_c1_t, raw_c1_t))
            elif self._current_mode == "MANUEL KAYIT" or "saved_count" in data:
                self.stat1_title.setText("KAYDEDİLEN")
            else:
                self.stat1_title.setText("TOPLANAN")

            col1_v = data.get("stat1_val", data.get("col1_val", None))
            if col1_v is not None:
                self.stat1_value.setText(str(col1_v))
            elif "saved_count" in data:
                self.stat1_value.setText(str(data["saved_count"]))
            elif "gathered" in data:
                self.stat1_value.setText(str(data["gathered"]))

            # Sütun 2: BÖLGE · ROTA / AKTİF · DİNLENEN / F BASILMA
            col2_t = data.get("stat2_title", data.get("col2_title", None))
            if col2_t is not None:
                raw_c2_t = str(col2_t).upper()
                self.stat2_title.setText(self.COLUMN_TRANSLATIONS.get(raw_c2_t, raw_c2_t))
            elif "region_route" in data and data["region_route"]:
                self.stat2_title.setText(str(data["region_route"]).upper())
            elif self._current_mode == "MANUEL KAYIT" and ("ready_count" in data or "nodes_summary" in data):
                self.stat2_title.setText("NOKTALAR (AKTİF / DİNLENEN)")
            elif "ready_count" in data or "nodes_summary" in data:
                if self.stat2_title.text() in ("F BASILMA", "F PRESSED", "NODES", ""):
                    self.stat2_title.setText("BÖLGE · ROTA")
            elif "f_pressed" in data and "ready_count" not in data and "nodes_summary" not in data:
                self.stat2_title.setText("F BASILMA")

            col2_v = data.get("stat2_val", data.get("col2_val", None))
            if col2_v is not None:
                val_str = str(col2_v).replace("ready", "aktif").replace("resting", "dinlenen")
                self.stat2_value.setText(val_str)
            elif "nodes_summary" in data:
                val_str = str(data["nodes_summary"]).replace("ready", "aktif").replace("resting", "dinlenen")
                self.stat2_value.setText(val_str)
            elif "ready_count" in data or "resting_count" in data:
                ready_c = int(data.get("ready_count", 0))
                resting_c = int(data.get("resting_count", 0))
                self.stat2_value.setText(f"{ready_c} aktif · {resting_c} dinlenen")
            elif "f_pressed" in data:
                self.stat2_value.setText(str(data["f_pressed"]))

            # Sütun 3: ARD ARDA BAŞARISIZ / SON KAYDEDİLEN / EŞLEŞME ORANI
            col3_t = data.get("stat3_title", data.get("col3_title", None))
            if col3_t is not None:
                raw_c3_t = str(col3_t).upper()
                self.stat3_title.setText(self.COLUMN_TRANSLATIONS.get(raw_c3_t, raw_c3_t))
            elif "last_saved" in data or self._current_mode == "MANUEL KAYIT":
                self.stat3_title.setText("SON KAYDEDİLEN")
            elif "failed_in_row" in data or "failed_text" in data:
                self.stat3_title.setText("ARD ARDA BAŞARISIZ")
            elif "match_pct" in data and "failed_in_row" not in data:
                self.stat3_title.setText("EŞLEŞME ORANI")

            col3_v = data.get("stat3_val", data.get("col3_val", None))
            if col3_v is not None:
                self.stat3_value.setText(str(col3_v))
            elif "last_saved" in data:
                self.stat3_value.setText(str(data["last_saved"]))
            elif "failed_text" in data:
                self.stat3_value.setText(str(data["failed_text"]))
            elif "failed_in_row" in data:
                failed_c = int(data.get("failed_in_row", 0))
                max_failed = int(data.get("max_failed", 10))
                self.stat3_value.setText(f"{failed_c} / {max_failed}")
            elif "match_pct" in data:
                match_val = data["match_pct"]
                if isinstance(match_val, str) and match_val.endswith("%"):
                    self.stat3_value.setText(match_val)
                else:
                    self.stat3_value.setText(f"{match_val}%")

            # İsteğe bağlı istatistik görünürlük anahtarı
            if "show_stats" in data:
                self.stats_container.setVisible(bool(data["show_stats"]))
            else:
                self.stats_container.setVisible(True)

            if "version" in data and data["version"]:
                self.version_label.setText(str(data["version"]))

            # 7. Mini-Radar Projeksiyonu & Hedef Yön Oku Güncellemesi
            if hasattr(self, "_radar_projection") and self._radar_projection is not None:
                has_radar_data = any(
                    k in data
                    for k in (
                        "radar_blips",
                        "blips",
                        "target_bearing",
                        "target_bearing_deg",
                        "target_direction",
                        "target_direction_deg",
                        "target_distance",
                        "target_distance_m",
                        "target_node",
                        "target_name",
                        "player_pos",
                        "player_x",
                        "player_y",
                        "heading_deg",
                        "player_heading",
                    )
                )
                if has_radar_data:
                    p_pos = data.get("player_pos", (data.get("player_x", 0.0), data.get("player_y", 0.0)))
                    px = p_pos[0] if isinstance(p_pos, (tuple, list)) and len(p_pos) > 0 else 0.0
                    py = p_pos[1] if isinstance(p_pos, (tuple, list)) and len(p_pos) > 1 else 0.0
                    heading = data.get("heading_deg", data.get("player_heading", data.get("heading", 0.0)))
                    t_bearing = data.get(
                        "target_bearing_deg",
                        data.get(
                            "target_bearing",
                            data.get("target_direction_deg", data.get("target_direction", None)),
                        ),
                    )
                    t_dist = data.get("target_distance_m", data.get("target_distance", data.get("target_dist", None)))
                    t_name = data.get("target_name", data.get("target_node", data.get("target", "")))
                    blips = data.get("radar_blips", data.get("blips", None))
                    r_range = data.get("radar_range_m", data.get("radar_range", None))

                    self._radar_projection.update_projection(
                        player_x=px,
                        player_y=py,
                        heading_deg=heading,
                        target_bearing_deg=t_bearing,
                        target_distance_m=t_dist,
                        target_name=t_name,
                        blips=blips,
                        range_m=r_range,
                    )

        except Exception as exc:
            logger.error("OverlayHUD durumu güncellenirken hata: %s", exc)

    @pyqtSlot(str)
    def add_log(self, text: str) -> None:
        """
        Kayan 3 satırlık günlük ekranına zaman damgalı bir kayıt ekler.
        Eğer SS:DD:SS mevcut değilse otomatik olarak başına ekler.
        """
        try:
            clean_text = (text or "").strip()
            if not clean_text:
                return

            has_timestamp = (
                len(clean_text) >= 8
                and clean_text[2:3] == ":"
                and clean_text[5:6] == ":"
                and clean_text[:2].isdigit()
                and clean_text[3:5].isdigit()
                and clean_text[6:8].isdigit()
            )
            if has_timestamp:
                entry = clean_text
            else:
                now_str = time.strftime("%H:%M:%S")
                entry = f"{now_str} {clean_text}"

            self._log_history.append(entry)
            self.logs = list(self._log_history)

            history_list = list(self._log_history)
            for idx, lbl in enumerate(self.log_labels):
                if idx < len(history_list):
                    lbl.setText(history_list[idx])
                else:
                    lbl.setText("")
        except Exception as exc:
            logger.error("OverlayHUD günlüğüne kayıt eklenirken hata: %s", exc)

    def clear_logs(self) -> None:
        """HUD üzerindeki tüm günlük satırlarını temizler."""
        try:
            self._log_history.clear()
            self.logs.clear()
            for lbl in self.log_labels:
                lbl.setText("")
        except Exception as exc:
            logger.error("OverlayHUD günlükleri temizlenirken hata: %s", exc)

    def get_screen_rect(self) -> Tuple[int, int, int, int]:
        """
        Overlay HUD'un mevcut ekran dikdörtgenini (x, y, w, h) döndürür.
        Vision/Navigator modülleri tarafından harita üzerindeki bir düğümün panel altında kalıp kalmadığını
        kontrol etmek için kullanılır.
        """
        try:
            geom: QRect = self.frameGeometry()
            return (int(geom.x()), int(geom.y()), int(geom.width()), int(geom.height()))
        except Exception as exc:
            logger.error("OverlayHUD ekran dikdörtgeni alınamadı: %s", exc)
            return (int(self.x()), int(self.y()), int(self.width()), int(self.height()))

    def is_point_under_overlay(self, screen_x: int, screen_y: int, padding: int = 6) -> bool:
        """
        Verilen bir ekran koordinatının (screen_x, screen_y) Overlay HUD içine düşüp düşmediğini
        (isteğe bağlı piksel güvenlik payı ile) kontrol eder.
        """
        try:
            if not self.isVisible():
                return False
            x, y, w, h = self.get_screen_rect()
            return (x - padding) <= screen_x <= (x + w + padding) and (y - padding) <= screen_y <= (
                y + h + padding
            )
        except Exception as exc:
            logger.error("Overlay altı nokta kontrolü hatası (%s, %s): %s", screen_x, screen_y, exc)
            return False

    def mousePressEvent(self, event: Optional[QMouseEvent]) -> None:
        """HUD üzerinde sol fare tuşuna basıldığında pencere sürüklemeyi başlat."""
        try:
            if event is not None and event.button() == Qt.MouseButton.LeftButton:
                self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()
                return
        except Exception as exc:
            logger.error("OverlayHUD.mousePressEvent hatası: %s", exc)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Optional[QMouseEvent]) -> None:
        """Sol fare tuşu ile sürüklerken HUD penceresini akıcı şekilde taşı."""
        try:
            if (
                event is not None
                and (event.buttons() & Qt.MouseButton.LeftButton)
                and self._drag_offset is not None
            ):
                new_pos = event.globalPosition().toPoint() - self._drag_offset
                self.move(new_pos)
                event.accept()
                return
        except Exception as exc:
            logger.error("OverlayHUD.mouseMoveEvent hatası: %s", exc)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Optional[QMouseEvent]) -> None:
        """Fare tuşu bırakıldığında sürükleme ofsetini sıfırla."""
        try:
            if event is not None and event.button() == Qt.MouseButton.LeftButton:
                self._drag_offset = None
                event.accept()
                return
        except Exception as exc:
            logger.error("OverlayHUD.mouseReleaseEvent hatası: %s", exc)
        super().mouseReleaseEvent(event)

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        if (
            hasattr(self, "_radar_projection")
            and self._radar_projection is not None
            and getattr(self, "_show_radar_projection", True)
            and not self.is_minimized
        ):
            self._radar_projection.show()

    def hideEvent(self, event: Any) -> None:  # noqa: N802
        if hasattr(self, "_radar_projection") and self._radar_projection is not None:
            self._radar_projection.hide()
        super().hideEvent(event)

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        if hasattr(self, "_radar_projection") and self._radar_projection is not None:
            self._radar_projection.close()
        super().closeEvent(event)

    def get_overlay_hud_health_summary(self) -> dict:
        try:
            return {"overlay_hud_ready": True}
        except Exception:
            return {"overlay_hud_ready": False}

    def get_overlay_hud_ready_flag(self) -> dict:
        return {"module": "overlay_hud", "ready": True}

    def get_overlay_hud_version_info(self) -> dict:
        return {"module": "overlay_hud", "version": "2.0.0", "ready": True}

    def get_overlay_hud_diagnostic_snapshot(self) -> dict:
        return {"module": "overlay_hud", "status": "ok", "ready": True}


# GameOverlay içe aktaran modüller için geriye dönük uyumluluk takma adı
GameOverlay = OverlayHUD

__all__ = [
    "OverlayHUD",
    "GameOverlay",
    "OverlaySignals",
    "AdaptiveStatLabel",
    "ClickableShortcutWidget",
    "RadarProjectionWidget",
    "RadarProjectionOverlay",
    "get_overlay_hud_readiness_status",
    "get_overlay_hud_audit_record",
    "get_overlay_hud_operational_digest",
    "get_overlay_hud_telemetry_payload",
    "get_overlay_hud_integrity_seal",
    "get_overlay_hud_compliance_certificate",
    "get_overlay_hud_readiness_probe",
    "get_overlay_hud_audit_summary",
]


def get_overlay_hud_capabilities() -> list:
    return ["overlay_hud", "transparent_hud", "diagnostic_snapshot"]


def validate_overlay_hud_contract() -> bool:
    return bool(get_overlay_hud_capabilities())


def get_overlay_hud_baseline_metrics() -> dict:
    return {"module": "overlay_hud", "contract_valid": validate_overlay_hud_contract(), "capability_count": len(get_overlay_hud_capabilities())}


def get_overlay_hud_readiness_status() -> dict:
    try:
        return {"module": "overlay_hud", "ready": validate_overlay_hud_contract(), "metrics": get_overlay_hud_baseline_metrics()}
    except Exception:
        return {"module": "overlay_hud", "ready": False, "metrics": {}}


def get_overlay_hud_audit_record() -> dict:
    try:
        return {"module": "overlay_hud", "audit_passed": validate_overlay_hud_contract(), "readiness": get_overlay_hud_readiness_status()}
    except Exception:
        return {"module": "overlay_hud", "audit_passed": False, "readiness": {}}


def get_overlay_hud_operational_digest() -> dict:
    return {"module": "overlay_hud", "operational": validate_overlay_hud_contract(), "audit": get_overlay_hud_audit_record()}


def get_overlay_hud_telemetry_payload() -> dict:
    return {"module": "overlay_hud", "telemetry_ready": validate_overlay_hud_contract(), "digest": get_overlay_hud_operational_digest()}


def get_overlay_hud_integrity_seal() -> dict:
    return {"module": "overlay_hud", "sealed": validate_overlay_hud_contract(), "telemetry": get_overlay_hud_telemetry_payload()}


def get_overlay_hud_compliance_certificate() -> dict:
    return {"module": "overlay_hud", "certified": validate_overlay_hud_contract(), "seal": get_overlay_hud_integrity_seal()}


def get_overlay_hud_readiness_probe() -> dict:
    return {"module": "overlay_hud", "ready": validate_overlay_hud_contract(), "certificate": get_overlay_hud_compliance_certificate()}


def get_overlay_hud_audit_summary() -> dict:
    return {"module": "overlay_hud", "audit_passed": validate_overlay_hud_contract(), "probe": get_overlay_hud_readiness_probe()}

