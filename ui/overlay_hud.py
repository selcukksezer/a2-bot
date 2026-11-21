# -*- coding: utf-8 -*-
"""
PyQt6 Oyun İçi Şeffaf Durum Paneli (Overlay HUD — Türkçe arayüz)
Gatherion-tarzı Overlay HUD'un piksel hassasiyetinde Türkçe uygulaması:
- Çerçevesiz, Her Zaman Üstte, Yarı Şeffaf Koyu Arka Planlı Araç Penceresi & WS_EX_TRANSPARENT tıklama geçişi (click-through)
- Dinamik 3px sol vurgu kenarlığı & durum rozeti (YÜRÜYOR, TOPLANIYOR, SAVAŞTA, DURAKLATILDI/BEKLEMEDE, MOLA/HARİTA)
- Mod kapsül rozeti (ROTA MODU, MANUEL KAYIT, OTO KAYIT, OTO KEŞİF) & canlı geçen süre sayacı (MM:SS)
- Düğüm Başına 3x Toplama aşama rozeti (3x Toplama / Harvest: 1/3 -> 2/3 -> 3/3) ve canlı ilerleme çubuğu
- Sağ üst Minimap'ten otomatik kaydedilen canlı materyal sayacı (Haritalanan / Minimap Mapped: X)
- Ana eylem başlığı, açıklayıcı alt başlık ve ince 1px yatay ayırıcı çizgi
- 3 Sütunlu telemetri ızgarası (TOPLANAN/KAYDEDİLEN, BÖLGE·ROTA (AKTİF/DİNLENEN), ARD ARDA BAŞARISIZ/EŞLEŞME)
- Kayan 3 satırlık zaman damgalı etkinlik günlüğü akışı
- [F9] Menü, [F8] Nokta Ekle kısayol rozetleri, Haritalanan sayacı ve vX.Y.Z sürüm etiketi içeren etkileşimli alt bilgi
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

# F-0540 tek kaynak: rozet eşikleri config/schema.py sabitlerinden okunur
# (HP 65 / MP 45) — combat ile aynı değer, HUD %50'ye kadar READY göstermez.
from config.schema import HP_POTION_THRESHOLD_PCT, MP_POTION_THRESHOLD_PCT

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
    QCheckBox,
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

from version import __version__

logger = logging.getLogger(__name__)

# HUD sürümü — tek kaynak version.__version__; sürüm etiketi (vX.Y.Z)
# ve sürüm fonksiyonları buradan beslenir.
HUD_VERSION: str = __version__


def _safe_float(value: Any, default: float = 0.0) -> float:
    """None/bozuk sayısal değeri çökmeden varsayılanla değiştir (radar telemetrisi için)."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result

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
            layout.setSpacing(6)

            self.key_badge = QLabel(self._key_text, self)
            self.key_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.key_badge.setMinimumHeight(18)
            self.key_badge.setStyleSheet(
                """
                QLabel {
                    background-color: rgba(255, 255, 255, 0.08);
                    color: #eaf1f7;
                    border: 1px solid rgba(255, 255, 255, 0.16);
                    border-radius: 4px;
                    font-family: 'JetBrains Mono', 'Consolas', monospace;
                    font-size: 11px;
                    font-weight: 600;
                    padding: 1px 6px;
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
                    font-size: 12px;
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
            font_sz = 11
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
        self.show_blips: bool = True  # Prototip "Tespit kutuları" aç/kapa
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
        # Tarama zamanlayıcısı yalnızca widget görünürken çalışır (showEvent'te
        # başlar, hideEvent'te durur); gizliyken CPU harcanmaz.

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        if self._sweep_timer is not None and not self._sweep_timer.isActive():
            self._sweep_timer.start()

    def hideEvent(self, event: Any) -> None:  # noqa: N802
        if self._sweep_timer is not None and self._sweep_timer.isActive():
            self._sweep_timer.stop()
        super().hideEvent(event)

    def _on_sweep_tick(self) -> None:
        if not self.isVisible():
            return
        self._sweep_angle = (self._sweep_angle + 4.0) % 360.0
        self.update()

    @staticmethod
    def _normalize_blip(blip: Any) -> Optional[Dict[str, Any]]:
        """Tek blip kaydını doğrula; dx/dy verilmişse mesafe/yönü türet, bozuksa at."""
        if not isinstance(blip, dict):
            return None
        normalized = dict(blip)
        try:
            if ("distance_m" not in normalized and "dist" not in normalized) and (
                "dx" in normalized or "dy" in normalized
            ):
                dx = _safe_float(normalized.get("dx", 0.0))
                dy = _safe_float(normalized.get("dy", 0.0))
                normalized["distance_m"] = math.hypot(dx, dy)
            if ("bearing_deg" not in normalized and "bearing" not in normalized) and (
                "dx" in normalized or "dy" in normalized
            ):
                dx = _safe_float(normalized.get("dx", 0.0))
                dy = _safe_float(normalized.get("dy", 0.0))
                # Yön: 0° = Kuzey (yukarı, -dy), saat yönünde artar.
                normalized["bearing_deg"] = math.degrees(math.atan2(dx, -dy)) % 360.0
            # Varsa mevcut mesafe/yön değerleri de sayısal olmalı; bozuksa at.
            for key in ("distance_m", "dist", "bearing_deg", "bearing"):
                if key in normalized:
                    if normalized[key] is None:
                        del normalized[key]
                    else:
                        float(normalized[key])
            # Konum bilgisi hiç yoksa çizilecek bir şey yoktur.
            if not any(
                k in normalized
                for k in ("distance_m", "dist", "bearing_deg", "bearing", "dx", "dy")
            ):
                return None
        except (TypeError, ValueError):
            return None
        return normalized

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
        self.player_pos = (_safe_float(player_x), _safe_float(player_y))
        self.heading_deg = _safe_float(heading_deg) % 360.0
        self.target_bearing_deg = (
            _safe_float(target_bearing_deg) % 360.0 if target_bearing_deg is not None else None
        )
        self.target_distance_m = (
            _safe_float(target_distance_m) if target_distance_m is not None else None
        )
        self.target_name = str(target_name or "")
        if blips is not None:
            try:
                normalized_blips = [
                    b for b in (self._normalize_blip(b) for b in list(blips)) if b is not None
                ]
            except (TypeError, ValueError):
                logger.warning("RadarProjectionWidget: bozuk blip listesi yoksayıldı")
                normalized_blips = self.blips
            self.blips = normalized_blips
        if range_m is not None and _safe_float(range_m, -1.0) > 0:
            self.range_m = _safe_float(range_m)
        self.update()

    def set_blips(self, blips: List[Dict[str, Any]]) -> None:
        try:
            self.blips = [
                b for b in (self._normalize_blip(b) for b in list(blips)) if b is not None
            ]
        except (TypeError, ValueError):
            logger.warning("RadarProjectionWidget.set_blips: bozuk blip listesi yoksayıldı")
            return
        self.update()

    def set_target_direction(
        self,
        bearing_deg: float,
        distance_m: float,
        label: str = "",
    ) -> None:
        self.target_bearing_deg = _safe_float(bearing_deg) % 360.0
        self.target_distance_m = _safe_float(distance_m)
        self.target_name = str(label or "")
        self.update()

    def set_blips_visible(self, visible: bool) -> None:
        """Tespit kutusu bliplerini çiz/kapat (veri saklanır, yalnızca çizim durur)."""
        self.show_blips = bool(visible)
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
        # show_blips False ise veri korunur, yalnızca çizim atlanır.
        for b in self.blips if self.show_blips else []:
            if not isinstance(b, dict):
                continue
            try:
                dist = float(b.get("distance_m", b.get("dist", 0.0)))
                bearing = float(b.get("bearing_deg", b.get("bearing", 0.0)))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(dist) or not math.isfinite(bearing):
                continue
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
        readout_y = self.height() - 15
        painter.setFont(QFont("Consolas", 9, QFont.Weight.DemiBold))
        if self.target_bearing_deg is not None and self.target_distance_m is not None:
            text = f"Hedef: {self.target_distance_m:.1f}m | {int(self.target_bearing_deg):03d}°"
            painter.setPen(QPen(QColor(240, 166, 60, 235)))
        else:
            text = f"Radar {int(self.range_m)}m · {len(self.blips)} blip"
            painter.setPen(QPen(QColor(148, 163, 184, 180)))
        painter.drawText(
            QRectF(0, readout_y, self.width(), 15), Qt.AlignmentFlag.AlignCenter, text
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
        # Konum: 1920px varsayımı yok — birincil ekranın sağ üst köşesine yaslanır.
        try:
            app = QApplication.instance()
            screen = app.primaryScreen() if app is not None else None
            if screen is not None:
                geom = screen.availableGeometry()
                self.move(
                    max(geom.left(), geom.right() - self.width() - 20),
                    geom.top() + 40,
                )
            else:
                self.move(1660, 40)
        except Exception:
            self.move(1660, 40)
        self.setWindowTitle("NVIDIA Mini-Radar Projection")
        self.set_click_through(True)

    def set_click_through(self, enabled: bool = True) -> bool:
        self._click_through_enabled = bool(enabled)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, bool(enabled))
        if sys.platform == "win32":
            try:
                hwnd = int(self.winId()) if self.winId() is not None else 0
                if not hwnd:
                    return False
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
                # Uygulandığını doğrula: istenen bit gerçekten ayarlı mı?
                applied_style = int(user32.GetWindowLongW(hwnd, GWL_EXSTYLE))
                if enabled and not (applied_style & WS_EX_TRANSPARENT):
                    return False
                if not enabled and (applied_style & WS_EX_TRANSPARENT):
                    return False
            except Exception as exc:
                logger.debug("RadarProjectionOverlay set_click_through hatası: %s", exc)
                return False
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

    def set_blips_visible(self, visible: bool) -> None:
        self.widget.set_blips_visible(bool(visible))

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
    stop_clicked = pyqtSignal()  # Durdur only — never resume
    settings_clicked = pyqtSignal()  # Overlay gear → main Ayarlar (no resume)
    f8_clicked = pyqtSignal()
    quit_requested = pyqtSignal()
    harvest_stage_updated = pyqtSignal(int, int)
    minimap_mapped_updated = pyqtSignal(int)
    consumable_status_updated = pyqtSignal(dict)
    # A grubu devri (HUD/mobil panik tüketicisi): panik durumu değişince yayılır.
    panic_status_updated = pyqtSignal(dict)

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
        # Prototip görsel durumları: dinlenme (3x hasat / 5 başarısız sonrası)
        # ve genel "çalışıyor" — state machine bunları gönderirse rozet doğru çıkar.
        "RESTING": ("DİNLENİYOR", "#3fc1d4", "rgba(63,193,212,0.10)", "#3fc1d4"),
        "RESTING_BREAK": ("DİNLENİYOR", "#3fc1d4", "rgba(63,193,212,0.10)", "#3fc1d4"),
        "DİNLENİYOR": ("DİNLENİYOR", "#3fc1d4", "rgba(63,193,212,0.10)", "#3fc1d4"),
        "RUNNING": ("ÇALIŞIYOR", "#43d68a", "rgba(67,214,138,0.10)", "#43d68a"),
        "ÇALIŞIYOR": ("ÇALIŞIYOR", "#43d68a", "rgba(67,214,138,0.10)", "#43d68a"),
        "MAP": ("HARİTA", "#e9b949", "rgba(233,185,73,0.12)", "#f5cf6a"),
        "NAVIGATING": ("HARİTA", "#e9b949", "rgba(233,185,73,0.12)", "#f5cf6a"),
        "HARİTA": ("HARİTA", "#e9b949", "rgba(233,185,73,0.12)", "#f5cf6a"),
    }

    MODE_LABELS: Dict[str, str] = {
        "route": "ROTA MODU",
        "route_mode": "ROTA MODU",
        "ROUTE": "ROTA MODU",
        "ROTA MODU": "ROTA MODU",
        # State machine'in gerçek rozet adları (core/state_machine.py
        # MODE_BADGE_MAP): bunlar tanınmadan "bilinmeyen mod" uyarısı veriyordu.
        "TOPLAMA (ROUTE)": "TOPLAMA (ROUTE)",
        "gather": "TOPLAMA (ROUTE)",
        "topla": "TOPLAMA (ROUTE)",
        "SAVAŞ (COMBAT)": "SAVAŞ (COMBAT)",
        # Prototip combat chip'i "SAVAŞ (ROUTE)" yazar; gerçek rozet adına çevrilir.
        "SAVAŞ (ROUTE)": "SAVAŞ (COMBAT)",
        "combat": "SAVAŞ (COMBAT)",
        "combat_mode": "SAVAŞ (COMBAT)",
        "COMBAT": "SAVAŞ (COMBAT)",
        "SAVAŞ": "SAVAŞ (COMBAT)",
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
        "NODES": "NOKTALAR",
        "NODES (ACTIVE / RESTING)": "NOKTALAR (AKTİF / DİNLENEN)",
        "ACTIVE": "AKTİF",
        "ACTIVE / RESTING": "AKTİF / DİNLENEN",
        "RESTING": "DİNLENEN",
        "REGION": "BÖLGE",
        "ROUTE": "ROTA",
        "REGION · ROUTE": "BÖLGE · ROTA",
        "REGION / ROUTE": "BÖLGE / ROTA",
        "F PRESSED": "F BASILMA",
        "FAILED IN A ROW": "ARD ARDA BAŞARISIZ",
        "FAILED": "BAŞARISIZ",
        "LAST SAVED": "SON KAYDEDİLEN",
        "PROMPT MATCH": "EŞLEŞME ORANI",
        "MATCH": "EŞLEŞME",
        "DISTANCE": "MESAFE",
        "TARGET": "HEDEF",
        "SPEED": "HIZ",
        "STATUS": "DURUM",
        "TIME": "SÜRE",
        "ELAPSED": "GEÇEN SÜRE",
        "REMAINING": "KALAN",
        "PROGRESS": "İLERLEME",
        "SCORE": "SKOR",
        "LEVEL": "SEVİYE",
        "HP": "CAN",
        "MP": "MANA",
        "HEALTH": "CAN",
        "MANA": "MANA",
        "COUNT": "SAYI",
        "TOTAL": "TOPLAM",
        "NEXT NODE": "SONRAKİ NOKTA",
        "CURRENT NODE": "MEVCUT NOKTA",
    }

    @staticmethod
    def _translate_column(raw_title: str) -> str:
        """Sütun başlığını Türkçeye çevir; bileşik başlıklarda (' / ', ' · ')
        bilinen parçaları tek tek çevirir, bilinmeyen parça aynen kalır."""
        raw = (raw_title or "").strip().upper()
        if not raw:
            return raw
        direct = OverlayHUD.COLUMN_TRANSLATIONS.get(raw)
        if direct is not None:
            return direct
        for sep in (" / ", " · "):
            if sep in raw:
                parts = [OverlayHUD.COLUMN_TRANSLATIONS.get(p.strip(), p.strip()) for p in raw.split(sep)]
                return sep.join(parts)
        return raw

    DEFAULT_SIZE: Tuple[int, int] = (760, 290)
    MINIMUM_SIZE: Tuple[int, int] = (720, 275)
    EXPANDED_SIZE: Tuple[int, int] = DEFAULT_SIZE
    MINIMIZED_SIZE: Tuple[int, int] = (52, 40)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._drag_offset: Optional[QPoint] = None
        self._session_start_time: float = time.monotonic()
        self._custom_elapsed_str: Optional[str] = None
        self._current_state: str = "WAITING"
        self._current_state_key: str = "WAITING"
        self._current_mode: str = "ROTA MODU"
        self._current_mode_key: str = ""
        self._ever_started: bool = False
        # Prototip (gatherion-hud-prototipi.html) canlı kaynakları: rota
        # ilerlemesi, hedef, sistem durumu — bilinmiyorsa "—" gösterilir.
        self._route_manager: Any = None
        self._worker: Any = None
        self._system_capture: Any = None
        self._route_current: Optional[int] = None
        self._route_total: Optional[int] = None
        self._route_loops: Optional[int] = None  # Canlı tur sayacı yok -> "—"
        self._route_zone: str = ""
        self._target_node_id: Any = None
        self._target_distance_m: Optional[float] = None
        self._target_material: str = ""
        self._system_status: Dict[str, Any] = {
            "fps": None,
            "gpu_pct": None,
            "cpu_pct": None,
            "gpu_name": None,
            "gpu_temp_c": None,
            "gpu_mem": None,
            "cpu_name": None,
            "cpu_freq_ghz": None,
        }
        self._detection_boxes_visible: bool = True
        self._minimap_visible: bool = True
        self._slow_tick: int = 0
        self._accent_color: str = "#64748b"
        self._log_history: Deque[str] = deque(maxlen=3)
        self.logs: list[str] = []

        # Düğüm Başına 3x Toplama & Sağ Üst Minimap Oto-Kayıt durumu
        self._current_harvest_stage: int = 0
        self._max_harvest_stages: int = 3
        # K-1: aşama henüz hiçbir canlı kaynaktan gelmediyse rozet "0/3"
        # diye yalan söylemez — bilinmiyor "—/3" kalır. İlk gerçek aşama
        # update_state/update_harvest_stage ile gelince True olur.
        self._harvest_stage_known: bool = False
        # A grubu devri: canlı panik kilidi (PanicAlertSystem) + son gösterilen
        # durum imzası (aynı durum tekrar tekrar çizilmez/loglanmaz).
        self._panic_lock: Any = None
        self._panic_signature: Optional[str] = None
        self._panic_label: Any = None
        self._minimap_mapped_count: int = 0
        self._learned_frames: int = 0
        # HP/MP başlangıçta BİLİNMİYOR (None): %100 varsayımı uydurma olur.
        # Gerçek değer ilk canlı vital güncellemesiyle gelir (update_state ->
        # update_consumable_and_selective_status).
        self._hp_pct: Optional[float] = None
        self._mp_pct: Optional[float] = None
        self._active_scrolls: int = 0
        self._selective_target: str = ""
        self._selective_enabled: bool = False
        self._selective_materials: List[str] = []
        self._consumable_status: Dict[str, Any] = {
            "hp_pct": None,
            "mp_pct": None,
            "active_scrolls": 0,
            "selective_target": "",
            "selective_active": False,
            "hp_potion": "BİLİNMİYOR",
            "mp_potion": "BİLİNMİYOR",
            "buff_scroll": "BİLİNMİYOR",
            "consumables_ready": False,
            "badge_text": "HP ?% · MP ?% · S:0",
            "selective_badge_text": "Seç-Topla: Tümü",
        }
        self._enterprise_status: Dict[str, Any] = {}
        self._f9_menu: Any = None
        # Seçici tıklama geçişi (Windows WM_NCHITTEST): boş alan oyun tıklamasını
        # geçirir, düğmeler tıklanabilir kalır. Qt WA_TransparentForMouseEvents
        # kullanılmaz — o bayrak düğmeleri de öldürür.
        self._click_through_enabled: bool = True
        self._stealth_ex_style: int = WS_EX_LAYERED | WS_EX_TOOLWINDOW
        self._minimized: bool = False
        self._ui_ready: bool = False
        self._ui_init_error: Optional[str] = None
        self._radar_projection: Optional[RadarProjectionOverlay] = None
        self.radar_projection: Optional[RadarProjectionOverlay] = None
        self._show_radar_projection: bool = True

        self._init_window_properties()
        self._init_ui()
        self._init_radar_projection()
        self._init_timer()
        self._apply_state_theme("WAITING")
        self._refresh_harvest_badge_style()
        self._init_signals()
        # A grubu devri: önceki çalışmadan kalan son panik (disk) varsa
        # açılışta bir kez göster; canlı kilit bağlanınca canlı değer alır.
        try:
            self.refresh_panic_from_disk()
        except Exception:
            pass

    def _init_signals(self) -> None:
        """İşçi iş parçacıklarından gelen OverlaySignals sinyallerini HUD
        güncelleme yöntemlerine bağlar (sinyal kapsayıcısı ölü kod değildir)."""
        try:
            self.signals = OverlaySignals()
            self.signals.update_signal.connect(self.update_state)
            self.signals.log_signal.connect(self.add_log)
            self.signals.harvest_stage_signal.connect(self.update_harvest_stage)
            self.signals.minimap_mapped_signal.connect(self.update_minimap_mapped)
            self.signals.consumable_status_signal.connect(self._on_consumable_status_signal)
        except Exception as exc:
            logger.error("OverlayHUD sinyal bağlantısı kurulamadı: %s", exc)

    @pyqtSlot(dict)
    def _on_consumable_status_signal(self, payload: Dict[str, Any]) -> None:
        """consumable_status_signal dict'ini güncelleme çağrısına çevirir."""
        if not isinstance(payload, dict):
            return
        # Eksik anahtar None olarak geçer: güncelleme fonksiyonu None'u
        # "değiştirme, mevcut değeri koru" sayar; %100 uydurulmaz.
        self.update_consumable_and_selective_status(
            hp_pct=payload.get("hp_pct"),
            mp_pct=payload.get("mp_pct"),
            active_scrolls=payload.get("active_scrolls", self._active_scrolls),
            selective_target=payload.get(
                "selective_target", payload.get("selected_material", self._selective_target)
            ),
        )

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
        yolo_provider: Optional[str] = None,
        kinah_per_hour: Optional[int] = None,
        active_char: Optional[str] = None,
    ) -> None:
        """
        Enterprise status (YOLO provider, Kinah/saat, aktif karakter) bilgisini
        HUD'da GÖRÜNÜR şekilde gösterir: alt başlığa yazar ve tooltip'i
        günceller; değerler self._enterprise_status altında saklanır.
        NOT [O4, F-0682]: varsayılanlar None'dur — bilinmeyen değer uydurulmaz
        ("CUDA"/0/"Main" varsayılanları kaldırıldı); None "BİLİNMİYOR" diye
        gösterilir. Canlı çağıranlar ölçülmüş değeri geçirir.
        """
        try:
            self.yolo_provider = str(yolo_provider) if yolo_provider is not None else None
            self.kinah_per_hour = int(kinah_per_hour) if kinah_per_hour is not None else None
            self.active_char = str(active_char) if active_char is not None else None
            self._enterprise_status = {
                "yolo_provider": self.yolo_provider,
                "kinah_per_hour": self.kinah_per_hour,
                "active_char": self.active_char,
            }

            kinah_txt = f"{self.kinah_per_hour:,}" if self.kinah_per_hour is not None else "BİLİNMİYOR"
            formatted_sub = (
                f"Char: {self.active_char or 'BİLİNMİYOR'} | "
                f"YOLO: {self.yolo_provider or 'BİLİNMİYOR'} | Kinah/h: {kinah_txt}"
            )
            if hasattr(self, "subtitle_label") and self.subtitle_label is not None:
                current_text = self.subtitle_label.text()
                if "YOLO:" not in current_text and "Char:" not in current_text:
                    base = current_text.strip()
                    visible = f"{base} · {formatted_sub}" if base else formatted_sub
                    self.subtitle_label.setText(visible)
                self.subtitle_label.setToolTip(formatted_sub)
        except Exception as exc:
            logger.error("set_enterprise_status error: %s", exc)

    def set_click_through(self, enabled: bool = True) -> bool:
        """
        Seçici tıklama geçişi: etkinse Windows'ta WM_NCHITTEST ile HUD'un boş
        alanları oyun tıklamasını geçirir; düğmeler/kısayollar tıklanabilir
        kalır. WA_TransparentForMouseEvents bilinçli kullanılmaz — o bayrak
        çocuk düğmeleri de öldürür. Win32 stilinde WS_EX_TRANSPARENT
        ayarlanmaz (tüm pencereyi kör eder); yalnızca NCHITTEST seçer.
        Linux'ta NCHITTEST yoktur: bayrak saklanır, geçiş uygulanamaz —
        bu durumda enabled=True istenirse False döner (S-1601: eskiden
        uygulanamayan geçişe rağmen True raporlanıyordu; HUD boş
        alanlarda oyun tıklamalarını yutmaya devam eder). enabled=False
        her platformda gerçekten uygulanabilir durumda sayılır.
        """
        try:
            self._click_through_enabled = bool(enabled)
            if enabled:
                target_style = WS_EX_LAYERED | WS_EX_TOOLWINDOW
            else:
                target_style = WS_EX_LAYERED | WS_EX_TOOLWINDOW
            self._stealth_ex_style = target_style
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

            # S-1601: win32 dışı platformda HUD'un kendi seçici geçişi yoktur.
            hud_applied = True
            if sys.platform != "win32" and enabled:
                logger.warning(
                    "OverlayHUD click-through bu platformda uygulanamaz "
                    "(WM_NCHITTEST yalnızca Windows'ta var): bayrak saklandı, "
                    "HUD boş alanlarda tıklamaları yutmaya devam edecek."
                )
                hud_applied = False

            if sys.platform == "win32":
                hwnd = int(self.winId()) if self.winId() is not None else 0
                if not hwnd:
                    return False
                user32 = ctypes.windll.user32
                cur_style = int(user32.GetWindowLongW(hwnd, GWL_EXSTYLE))
                # WS_EX_TRANSPARENT bitini her durumda temizle: seçim NCHITTEST'te.
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
                applied_style = int(user32.GetWindowLongW(hwnd, GWL_EXSTYLE))
                if applied_style & WS_EX_TRANSPARENT:
                    return False
            if hasattr(self, "_radar_projection") and self._radar_projection is not None:
                try:
                    if not self._radar_projection.set_click_through(enabled):
                        return False
                except Exception:
                    pass
            return hud_applied
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
        button.setFixedSize(28, 24)
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
            mini_layout.setContentsMargins(8, 5, 8, 5)
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
            card_layout.setContentsMargins(12, 4, 12, 6)
            card_layout.setSpacing(1)

            # -----------------------------------------------------------------
            # 1. ÜST SATIR: Durum Rozeti | Mod | 3x Toplama | Esnek | Süre | Küçült
            # -----------------------------------------------------------------
            header_layout = QHBoxLayout()
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(6)

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
                    font-size: 12px;
                    font-weight: 700;
                    letter-spacing: 0.05em;
                    padding: 1px 6px;
                }
                """
            )

            # Düğüm Başına 3x Toplama aşama rozeti (ör. "3x Toplama (Harvest): 2/3")
            # K-1: başlangıç "0/3" değil "—/3" — aşama bilinmiyor.
            self.harvest_badge_label = QLabel("3x Toplama (Harvest: —/3)", self.card)
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
                    font-size: 12px;
                    font-weight: 700;
                    letter-spacing: 0.05em;
                    padding: 1px 6px;
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
                    font-size: 12px;
                    font-weight: 600;
                    padding: 1px 7px;
                }
                """
            )
            self.time_lbl = self.elapsed_label  # Geriye dönük uyumluluk takma adı

            self.btn_hud_minimize = self._make_hud_toggle_button(self.card, "HudMinimizeButton")
            self.btn_hud_minimize.setText("−")
            self.btn_hud_minimize.setToolTip("HUD'u küçült")

            # Prototip "HUD AYARLARI" paneli aç/kapa (Menü/F9 bağlantısına dokunmaz).
            # _make_hud_toggle_button KULLANILMAZ: o yardımcı küçültmeye bağlar.
            self.btn_hud_settings = QPushButton("⚙", self.card)
            self.btn_hud_settings.setObjectName("HudSettingsButton")
            self.btn_hud_settings.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            self.btn_hud_settings.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.btn_hud_settings.setFixedSize(26, 24)
            self.btn_hud_settings.setToolTip("Ayarlar")
            self.btn_hud_settings.setStyleSheet(
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
            self.btn_hud_settings.clicked.connect(self._on_settings_clicked)

            header_layout.addWidget(self.badge_label)
            header_layout.addWidget(self.mode_label)
            header_layout.addWidget(self.harvest_badge_label)
            header_layout.addStretch(1)
            header_layout.addWidget(self.elapsed_label)
            header_layout.addWidget(self.btn_hud_settings)
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
                    font-size: 15px;
                    font-weight: 700;
                    letter-spacing: 0.01em;
                    padding-top: 0px;
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
                    font-size: 11px;
                    font-weight: 500;
                    padding-bottom: 0px;
                }
                """
            )
            self.sub_lbl = self.subtitle_label  # Geriye dönük uyumluluk takma adı
            card_layout.addWidget(self.subtitle_label)

            # A grubu devri: panik kilit şeridi — kilit yokken gizli, kilit
            # aktifken kırmızı şerit + sebep; disk kaydından gelen son
            # panik (yeniden başlatma sonrası) nötr gri gösterilir.
            self.panic_label = QLabel("", self.card)
            self.panic_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.panic_label.setWordWrap(True)
            self.panic_label.setVisible(False)
            self._panic_label = self.panic_label
            card_layout.addWidget(self.panic_label)

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
            stats_grid.setHorizontalSpacing(16)
            stats_grid.setVerticalSpacing(1)

            self.stat1_title = QLabel("TOPLANAN", self.stats_container)
            self.stat1_value = QLabel("Toplanan: 0", self.stats_container)

            self.stat2_title = QLabel("AKTİF / DİNLENEN / ROTA", self.stats_container)
            self.stat2_value = AdaptiveStatLabel("0 aktif · 0 dinlenen", self.stats_container, base_size=16)

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
                    font-size: 16px;
                    font-weight: 700;
                }
            """

            for header_lbl in (self.stat1_title, self.stat2_title, self.stat3_title):
                header_lbl.setStyleSheet(header_col_style)
                header_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            for val_lbl in (self.stat1_value, self.stat3_value):
                val_lbl.setStyleSheet(value_col_style)
                val_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                val_lbl.setMinimumHeight(26)

            self.stat2_value.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.stat2_value.setMinimumHeight(26)
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
            # 3b. PROTOTİP ROTA SATIRI: "Nokta x/N · Tur n" + ilerleme çubuğu
            # Kaynak: bağlı route_manager + worker.active_node (canlı); tur
            # sayacı worker'da yok -> "—" (sahte tur numarası üretilmez).
            # -----------------------------------------------------------------
            self.route_row = QWidget(self.card)
            route_layout = QHBoxLayout(self.route_row)
            route_layout.setContentsMargins(0, 0, 0, 0)
            route_layout.setSpacing(6)
            self.route_label = QLabel("Rota: — · Nokta —/— · Tur —", self.route_row)
            self.route_label.setStyleSheet(
                """
                QLabel {
                    background: transparent;
                    color: #9fb2c3;
                    border: none;
                    font-family: 'Barlow', 'Segoe UI', sans-serif;
                    font-size: 11px;
                    font-weight: 600;
                }
                """
            )
            self.route_progress_bar = QProgressBar(self.route_row)
            self.route_progress_bar.setRange(0, 1)
            self.route_progress_bar.setValue(0)
            self.route_progress_bar.setTextVisible(False)
            self.route_progress_bar.setFixedHeight(4)
            self.route_progress_bar.setStyleSheet(
                """
                QProgressBar {
                    background-color: rgba(255, 255, 255, 0.10);
                    border: none;
                    border-radius: 3px;
                }
                QProgressBar::chunk {
                    background-color: #38bdf8;
                    border-radius: 3px;
                }
                """
            )
            route_layout.addWidget(self.route_label)
            route_layout.addWidget(self.route_progress_bar, 1)
            card_layout.addWidget(self.route_row)

            # -----------------------------------------------------------------
            # 3c. PROTOTİP HEDEF SATIRI: "Hedef: Nokta 3 · <materyal> · 120 m"
            # Kaynak: worker.active_node + active_node_distance_m (canlı);
            # hedef yokken "Hedef: —".
            # -----------------------------------------------------------------
            self.target_label = QLabel("Hedef: —", self.card)
            self.target_label.setStyleSheet(
                """
                QLabel {
                    background: transparent;
                    color: #eaf1f7;
                    border: none;
                    font-family: 'Barlow', 'Segoe UI', sans-serif;
                    font-size: 11px;
                    font-weight: 600;
                }
                """
            )
            card_layout.addWidget(self.target_label)

            # -----------------------------------------------------------------
            # 3d. SİSTEM DURUMU: FPS / GPU / CPU — yalnızca ölçülmüş değer.
            # Kaynak: core.system_monitor.SystemMonitor (GPU/CPU gerçek ölçüm;
            # yoksa "—"); FPS yalnızca capture örneği bağlandıysa ölçülür.
            # -----------------------------------------------------------------
            self.system_label = QLabel("FPS — · GPU — · CPU —", self.card)
            self.system_label.setToolTip("Sistem durumu: ölçüm yokken '—' gösterilir")
            self.system_label.setStyleSheet(
                """
                QLabel {
                    background: transparent;
                    color: #6d8093;
                    border: none;
                    font-family: 'JetBrains Mono', 'Consolas', monospace;
                    font-size: 11px;
                    font-weight: 500;
                }
                """
            )
            card_layout.addWidget(self.system_label)

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
            self.logs_container.setMinimumHeight(48)
            logs_layout = QVBoxLayout(self.logs_container)
            logs_layout.setContentsMargins(10, 2, 10, 2)
            logs_layout.setSpacing(1)

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
                log_lbl.setFixedHeight(14)
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
                "font-size: 11px;"
                "font-weight: 600;"
                "padding: 0px 6px;"
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
            action_layout.setContentsMargins(0, 0, 0, 0)
            action_layout.setSpacing(4)

            def action_button(text: str, name: str) -> QPushButton:
                button = QPushButton(text, self._action_row)
                button.setObjectName(name)
                button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                button.setFixedHeight(24)
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
            self.btn_menu.setToolTip("Menü (F9 veya Ctrl+Alt+F9)")
            self.btn_node.setToolTip("Nokta kaydet (Ctrl+Alt+F8)")
            self.btn_resume.setToolTip("Devam: Toplama modu")
            self.btn_combat.setToolTip("Savaş modu")
            self.btn_gather.setToolTip("Toplam: Toplama modu")
            self.btn_quit.setToolTip("Uygulamayı kapatır")
            self.btn_run.clicked.connect(self._on_f10_clicked)
            self.btn_stop.clicked.connect(self._on_stop_clicked)
            self.btn_menu.clicked.connect(self._on_f9_clicked)
            self.btn_node.clicked.connect(self._on_f8_clicked)
            self.btn_resume.clicked.connect(lambda _checked=False: self._dispatch_f9_action("resume"))
            self.btn_combat.clicked.connect(lambda _checked=False: self._dispatch_f9_action("combat"))
            self.btn_gather.clicked.connect(lambda _checked=False: self._dispatch_f9_action("route"))
            self.btn_quit.clicked.connect(self._on_quit_clicked)
            card_layout.addWidget(self._action_row)

            # -----------------------------------------------------------------
            # 4c. PROTOTİP "HUD AYARLARI" PANELİ (menuPanel karşılığı; gizli başlar,
            # başlıktaki ⚙ düğmesiyle açılır). Canlı kaynağı olan iki anahtar
            # gerçekten bağlanır (Mini harita -> radar penceresi; Tespit
            # kutuları -> radar blipleri). Diğer üçünün HUD kapsamında
            # okunur/yazılır canlı kaynağı YOKTUR (config şemasında karşılık
            # anahtar yok; worker'da değiştirilebilir öznitelik yok) — bu
            # yüzden devre dışı bırakılır, sahte anahtar gibi davranmazlar.
            # -----------------------------------------------------------------
            self.settings_panel = QFrame(self.card)
            self.settings_panel.setObjectName("HudSettingsPanel")
            self.settings_panel.setStyleSheet(
                """
                QFrame#HudSettingsPanel {
                    background: rgba(0, 0, 0, 0.32);
                    border: 1px solid rgba(255, 255, 255, 0.06);
                    border-radius: 8px;
                }
                QCheckBox {
                    background: transparent;
                    color: #eaf1f7;
                    font-family: 'Barlow', 'Segoe UI', sans-serif;
                    font-size: 12px;
                    font-weight: 600;
                }
                QCheckBox:disabled {
                    color: #6d8093;
                }
                """
            )
            settings_layout = QGridLayout(self.settings_panel)
            settings_layout.setContentsMargins(8, 6, 8, 6)
            settings_layout.setHorizontalSpacing(12)
            settings_layout.setVerticalSpacing(4)

            settings_title = QLabel("HUD AYARLARI", self.settings_panel)
            settings_title.setStyleSheet(
                "QLabel { background: transparent; color: #6d8093; border: none;"
                " font-family: 'Barlow', 'Segoe UI', sans-serif; font-size: 11px;"
                " font-weight: 700; letter-spacing: 0.16em; }"
            )
            settings_layout.addWidget(settings_title, 0, 0, 1, 2)

            self.chk_auto_route = QCheckBox("Otomatik rota", self.settings_panel)
            self.chk_auto_route.setToolTip(
                "HUD'dan değiştirilemez: bu ayarın canlı kaynağı yok"
                " (config şemasında karşılık anahtar yok); Ayarlar penceresinden yönetilir"
            )
            self.chk_auto_route.setEnabled(False)
            settings_layout.addWidget(self.chk_auto_route, 1, 0)

            self.chk_low_hp_escape = QCheckBox("Düşük CAN kaçışı", self.settings_panel)
            self.chk_low_hp_escape.setToolTip(
                "HUD'dan değiştirilemez: bu ayarın canlı kaynağı yok"
                " (worker'da aç/kapa özniteliği yok); Ayarlar penceresinden yönetilir"
            )
            self.chk_low_hp_escape.setEnabled(False)
            settings_layout.addWidget(self.chk_low_hp_escape, 1, 1)

            self.chk_sound_alert = QCheckBox("Sesli uyarı", self.settings_panel)
            self.chk_sound_alert.setToolTip(
                "HUD'dan değiştirilemez: bu ayarın canlı kaynağı yok"
                " (config şemasında ses anahtarı yok); Ayarlar penceresinden yönetilir"
            )
            self.chk_sound_alert.setEnabled(False)
            settings_layout.addWidget(self.chk_sound_alert, 2, 0)

            self.chk_minimap = QCheckBox("Mini harita", self.settings_panel)
            self.chk_minimap.setToolTip("Sağ üstteki yuvarlak harita (radar penceresi)")
            self.chk_minimap.setChecked(True)
            self.chk_minimap.toggled.connect(
                lambda checked: self.set_minimap_visible(bool(checked))
            )
            settings_layout.addWidget(self.chk_minimap, 2, 1)

            self.chk_detection_boxes = QCheckBox("Tespit kutuları", self.settings_panel)
            self.chk_detection_boxes.setToolTip("Sahnedeki YZ tespit blipleri (radar üzerinde)")
            self.chk_detection_boxes.setChecked(True)
            self.chk_detection_boxes.toggled.connect(
                lambda checked: self.set_detection_boxes_visible(bool(checked))
            )
            settings_layout.addWidget(self.chk_detection_boxes, 3, 0)

            self.btn_reset_pos = QPushButton("HUD konumunu sıfırla", self.settings_panel)
            self.btn_reset_pos.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            self.btn_reset_pos.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.btn_reset_pos.setFixedHeight(26)
            self.btn_reset_pos.clicked.connect(self.reset_position)
            settings_layout.addWidget(self.btn_reset_pos, 3, 1)

            self.btn_clear_log = QPushButton("Günlüğü temizle", self.settings_panel)
            self.btn_clear_log.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            self.btn_clear_log.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.btn_clear_log.setFixedHeight(26)
            self.btn_clear_log.clicked.connect(self.clear_logs)
            settings_layout.addWidget(self.btn_clear_log, 4, 0, 1, 2)

            self.settings_panel.hide()
            card_layout.addWidget(self.settings_panel)

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
            self.f9_shortcut.setToolTip("F9 (klasik) veya Ctrl+Alt+F9: Hızlı Kontrol Menüsü")
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
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 0.06em;
                    padding: 3px 10px;
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
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 0.06em;
                    padding: 3px 10px;
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
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 0.04em;
                    padding: 3px 8px;
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
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 0.04em;
                    padding: 3px 8px;
                }
                """
            )
            self.selective_badge_label.setVisible(False)
            self.selective_target_label = self.selective_badge_label

            self.version_label = QLabel(f"v{HUD_VERSION}", self.card)
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

            # Prototip footer "Harita" kısayolu (hot--map): rota haritası ekranını
            # F9 menüsü üzerinden açar. Aksiyon satırına 9. buton olarak eklenmez
            # (tek satırda 8 buton sınırı — uzun etiketlerde taşma yapar).
            self.btn_map = QPushButton("🗺 Harita", self.card)
            self.btn_map.setObjectName("HudMapButton")
            self.btn_map.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            self.btn_map.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.btn_map.setToolTip("Rota haritasını aç (F9 menüsü → Harita)")
            self.btn_map.setStyleSheet(
                """
                QPushButton {
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
                QPushButton:hover {
                    background-color: rgba(63, 193, 212, 0.22);
                }
                """
            )
            self.btn_map.clicked.connect(lambda _checked=False: self._dispatch_f9_action("map"))

            footer_layout.addWidget(self.f10_shortcut)
            footer_layout.addWidget(self.f9_shortcut)
            footer_layout.addWidget(self.f8_shortcut)
            footer_layout.addWidget(self.btn_map)
            footer_layout.addWidget(self.ai_learned_label)
            footer_layout.addWidget(self.minimap_mapped_label)
            footer_layout.addWidget(self.consumable_badge_label)
            footer_layout.addWidget(self.selective_badge_label)
            footer_layout.addStretch(1)
            footer_layout.addWidget(self.version_label)

            card_layout.addLayout(footer_layout)
            self._ui_ready = True
            self._ui_init_error = None
        except Exception as exc:
            # Yarım kurulmuş HUD sessizce dönmesin: hata saklanır, sağlık
            # özeti hazır olmadığını bildirir, update_state erken döner.
            self._ui_ready = False
            self._ui_init_error = str(exc)
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
            # K-1: aşama bilinmiyorsa "0/3" yazılmaz — "—/3" kalır.
            if not self.__dict__.get("_harvest_stage_known", False):
                self.harvest_badge_label.setText(
                    f"3x Toplama (Harvest: —/{max(1, int(self._max_harvest_stages))})"
                )
                self.harvest_progress_bar.setRange(0, max(1, int(self._max_harvest_stages)))
                self.harvest_progress_bar.setValue(0)
                return
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
                    font-size: 12px;
                    font-weight: 700;
                    letter-spacing: 0.05em;
                    padding: 3px 6px;
                }}
                """
            )
            # Layout squishing'i önlemek için metin genişliği + dolguyu garantiye al
            fm = self.harvest_badge_label.fontMetrics()
            adv = fm.horizontalAdvance(self.harvest_badge_label.text())
            self.harvest_badge_label.setMinimumWidth(adv + 16)
        except Exception as exc:
            logger.error("3x Toplama rozet stili güncellenemedi: %s", exc)

    # ------------------------------------------------------------------
    # Prototip uyumu (gatherion-hud-prototipi.html): rota ilerlemesi,
    # hedef satırı, sistem durumu, HUD ayarları, kontrol senkronizasyonu.
    # Hepsi canlı kaynaktan beslenir; kaynak yokken "—" gösterilir.
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_node_id(node: Any) -> Optional[int]:
        """NodeData / dict / int biçimlerindeki aktif düğümden sayısal id çıkar."""
        try:
            if node is None:
                return None
            if isinstance(node, bool):
                return None
            if isinstance(node, int):
                return int(node)
            if isinstance(node, dict):
                raw = node.get("id", node.get("node_id"))
                return int(raw) if raw is not None else None
            raw = getattr(node, "id", None)
            if raw is None:
                raw = getattr(node, "node_id", None)
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _resolve_mode_label(raw_mode: str) -> Optional[str]:
        """Mod adını büyük/küçük harf duyarsız çöz; bilinmeyen -> None."""
        raw = (raw_mode or "").strip()
        if not raw:
            return None
        upper = raw.upper()
        for key, label in OverlayHUD.MODE_LABELS.items():
            if key.upper() == upper:
                return label
        return None

    def _mode_key_for(self, raw_mode: str, resolved: str) -> str:
        """Mod düğmesi vurgusu için anahtar: gather | combat | diğer."""
        probe = f"{raw_mode} {resolved}".upper()
        if "SAVAŞ" in probe or "COMBAT" in probe or "FIGHT" in probe:
            return "combat"
        if "TOPLAMA" in probe or "ROUTE" in probe or "ROTA" in probe or "GATHER" in probe:
            return "gather"
        return "other"

    def reset_position(self) -> None:
        """HUD'u ilk konumuna (30, 30) döndürür (prototip: 'HUD konumunu sıfırla')."""
        try:
            self.move(30, 30)
            self.add_log("HUD konumu sıfırlandı")
        except Exception as exc:
            logger.error("HUD konumu sıfırlanamadı: %s", exc)

    def toggle_settings_panel(self) -> None:
        self.set_settings_panel_visible(not self.settings_panel.isVisible())

    def set_settings_panel_visible(self, visible: bool) -> None:
        try:
            self.settings_panel.setVisible(bool(visible))
        except Exception as exc:
            logger.error("HUD ayar paneli açılamadı: %s", exc)

    def set_minimap_visible(self, visible: bool) -> None:
        """Prototip 'Mini harita' anahtarı: radar penceresini gösterir/gizler."""
        try:
            self._minimap_visible = bool(visible)
            self.show_radar_projection(bool(visible))
            chk = self.__dict__.get("chk_minimap")
            if chk is not None and chk.isChecked() != bool(visible):
                chk.setChecked(bool(visible))
            self.add_log(f"Mini harita: {'AÇIK' if visible else 'KAPALI'}")
        except Exception as exc:
            logger.error("Mini harita durumu değiştirilemedi: %s", exc)

    def set_detection_boxes_visible(self, visible: bool) -> None:
        """Prototip 'Tespit kutuları' anahtarı: radar bliplerini çizer/gizler."""
        try:
            self._detection_boxes_visible = bool(visible)
            if self._radar_projection is not None:
                self._radar_projection.set_blips_visible(bool(visible))
            chk = self.__dict__.get("chk_detection_boxes")
            if chk is not None and chk.isChecked() != bool(visible):
                chk.setChecked(bool(visible))
        except Exception as exc:
            logger.error("Tespit kutuları durumu değiştirilemedi: %s", exc)

    def bind_system_monitor(self, capture: Any = None) -> None:
        """FPS ölçümü için yakalama örneğini bağlar (yoksa FPS '—' kalır)."""
        self._system_capture = capture
        try:
            self._refresh_system_status()
        except Exception as exc:
            logger.debug("Sistem durumu ilk okuma hatası: %s", exc)

    def update_route_progress(
        self,
        current: Optional[int] = None,
        total: Optional[int] = None,
        loops: Optional[int] = None,
        zone: Optional[str] = None,
    ) -> None:
        """Rota ilerlemesini açık değerlerle günceller (None = koru)."""
        try:
            if current is not None:
                self._route_current = max(0, int(current))
            if total is not None:
                self._route_total = max(0, int(total))
            if loops is not None:
                self._route_loops = max(0, int(loops))
            if zone is not None:
                self._route_zone = str(zone).strip()
            self._render_route_progress()
        except (TypeError, ValueError) as exc:
            logger.warning("Rota ilerlemesi güncellenemedi: %s", exc)

    def _render_route_progress(self) -> None:
        """Rota satırını + çubuğu çizer; bilinmeyen her alan '—' olur."""
        try:
            cur = self._route_current
            tot = self._route_total
            loops = self._route_loops
            zone = self._route_zone or "—"
            cur_txt = str(cur) if cur is not None else "—"
            tot_txt = str(tot) if tot is not None else "—"
            loops_txt = str(loops) if loops is not None else "—"
            self.route_label.setText(f"Rota: {zone} · Nokta {cur_txt}/{tot_txt} · Tur {loops_txt}")
            if tot:
                self.route_progress_bar.setRange(0, max(1, int(tot)))
                self.route_progress_bar.setValue(min(int(cur or 0), int(tot)))
            else:
                self.route_progress_bar.setRange(0, 1)
                self.route_progress_bar.setValue(0)
        except Exception as exc:
            logger.error("Rota ilerlemesi çizilemedi: %s", exc)

    def _refresh_route_progress(self) -> None:
        """Bağlı route_manager + worker'dan rota ilerlemesini türetir.

        Toplam: route_manager.get_route_nodes() uzunluğu (aktif bölge/rotalar).
        Mevcut: worker.active_node id'sinin listedeki sırası (1-tabanlı).
        Tur: worker/route_manager'da tur sayacı YOKTUR -> '—' bırakılır.
        Bölge: route_manager.active_region.
        Kaynak yoksa veya okuma başarısızsa mevcut değerler korunur.
        """
        try:
            rm = getattr(self, "_route_manager", None)
            worker = getattr(self, "_worker", None)
            if rm is None and worker is None:
                return
            nodes: List[Any] = []
            if rm is not None and hasattr(rm, "get_route_nodes"):
                try:
                    nodes = list(rm.get_route_nodes() or [])
                except TypeError:
                    nodes = []
            total = len(nodes) if nodes else None
            if total is not None:
                self._route_total = total
            zone = None
            if rm is not None:
                zone = getattr(rm, "active_region", None)
                if not zone:
                    data = getattr(rm, "data", None)
                    if isinstance(data, dict):
                        zone = data.get("active_region")
            if zone:
                self._route_zone = str(zone)
            current = None
            if worker is not None:
                node_id = self._coerce_node_id(getattr(worker, "active_node", None))
                if node_id is not None and nodes:
                    for idx, node in enumerate(nodes):
                        if self._coerce_node_id(node) == node_id:
                            current = idx + 1
                            break
                    if current is None:
                        # Düğüm listede yoksa (farklı rota) sıra bilinmiyor.
                        current = None
            if current is not None:
                self._route_current = current
            self._render_route_progress()
        except Exception as exc:
            logger.debug("Rota ilerlemesi türetilemedi: %s", exc)

    def update_target(
        self,
        node_id: Any = None,
        distance_m: Any = None,
        material: Optional[str] = None,
    ) -> None:
        """Hedef satırını açık değerlerle günceller (None = koru)."""
        try:
            if node_id is not None:
                self._target_node_id = node_id
            if distance_m is not None:
                try:
                    self._target_distance_m = _safe_float(distance_m)
                except (TypeError, ValueError):
                    pass
            if material is not None:
                self._target_material = str(material).strip()
            self._render_target()
        except Exception as exc:
            logger.error("Hedef satırı güncellenemedi: %s", exc)

    def _render_target(self) -> None:
        """Hedef satırını çizer; hedef yokken 'Hedef: —'."""
        try:
            nid = self._target_node_id
            dist = self._target_distance_m
            mat = self._target_material
            if nid is None and not mat and dist is None:
                self.target_label.setText("Hedef: —")
                return
            parts = []
            if nid is not None:
                parts.append(f"Nokta {nid}")
            if mat:
                parts.append(mat)
            if dist is not None:
                parts.append(f"{dist:.0f} m" if float(dist) == int(float(dist)) else f"{dist:.1f} m")
            self.target_label.setText("Hedef: " + (" · ".join(parts) if parts else "—"))
        except Exception as exc:
            logger.error("Hedef satırı çizilemedi: %s", exc)

    def _refresh_target(self) -> None:
        """Bağlı worker'dan hedefi türetir (active_node + mesafe)."""
        try:
            worker = getattr(self, "_worker", None)
            if worker is None:
                return
            node = getattr(worker, "active_node", None)
            if node is None:
                return
            nid = self._coerce_node_id(node)
            if nid is not None:
                self._target_node_id = nid
            mat = ""
            try:
                mat = str(getattr(node, "name", "") or "").strip()
            except Exception:
                mat = ""
            if not mat and isinstance(node, dict):
                mat = str(node.get("name", node.get("material", "")) or "").strip()
            if mat:
                self._target_material = mat
            try:
                dist = getattr(worker, "active_node_distance_m", None)
                if dist is not None:
                    self._target_distance_m = _safe_float(dist)
            except (TypeError, ValueError):
                pass
            self._render_target()
        except Exception as exc:
            logger.debug("Hedef türetilemedi: %s", exc)

    def _refresh_live_derived(self) -> None:
        """Bağlı canlı kaynaklardan türetilen tüm satırları tazeler."""
        try:
            self._refresh_route_progress()
        except Exception:
            pass
        try:
            self._refresh_target()
        except Exception:
            pass

    def update_system_status(
        self,
        fps: Optional[float] = None,
        gpu_pct: Optional[float] = None,
        cpu_pct: Optional[float] = None,
        gpu_name: Optional[str] = None,
        gpu_temp_c: Optional[float] = None,
        gpu_mem: Optional[str] = None,
        cpu_name: Optional[str] = None,
        cpu_freq_ghz: Optional[float] = None,
    ) -> None:
        """Sistem satırını açık (ölçülmüş) değerlerle günceller (None = koru)."""
        try:
            status = self._system_status
            if fps is not None:
                status["fps"] = _safe_float(fps) if _safe_float(fps, -1.0) > 0 else None
            if gpu_pct is not None:
                status["gpu_pct"] = max(0.0, min(100.0, _safe_float(gpu_pct)))
            if cpu_pct is not None:
                status["cpu_pct"] = max(0.0, min(100.0, _safe_float(cpu_pct)))
            if gpu_name is not None:
                status["gpu_name"] = str(gpu_name)
            if gpu_temp_c is not None:
                status["gpu_temp_c"] = _safe_float(gpu_temp_c)
            if gpu_mem is not None:
                status["gpu_mem"] = str(gpu_mem)
            if cpu_name is not None:
                status["cpu_name"] = str(cpu_name)
            if cpu_freq_ghz is not None:
                status["cpu_freq_ghz"] = _safe_float(cpu_freq_ghz)
            self._render_system_status()
        except (TypeError, ValueError) as exc:
            logger.warning("Sistem durumu güncellenemedi: %s", exc)

    def _render_system_status(self) -> None:
        """Sistem satırını çizer; ölçülmemiş her alan '—' olur."""
        try:
            status = self._system_status
            fps = status.get("fps")
            gpu = status.get("gpu_pct")
            cpu = status.get("cpu_pct")
            fps_txt = f"{fps:.0f}" if fps is not None else "—"
            gpu_txt = f"%{gpu:.0f}" if gpu is not None else "—"
            cpu_txt = f"%{cpu:.0f}" if cpu is not None else "—"
            self.system_label.setText(f"FPS {fps_txt} · GPU {gpu_txt} · CPU {cpu_txt}")
            tips = []
            if status.get("gpu_name"):
                gpu_tip = str(status["gpu_name"])
                if status.get("gpu_temp_c") is not None:
                    gpu_tip += f" · {status['gpu_temp_c']:.0f}°C"
                if status.get("gpu_mem"):
                    gpu_tip += f" · {status['gpu_mem']}"
                tips.append(f"GPU: {gpu_tip}")
            if status.get("cpu_name"):
                cpu_tip = str(status["cpu_name"])
                if status.get("cpu_freq_ghz") is not None:
                    cpu_tip += f" · {status['cpu_freq_ghz']:.1f} GHz"
                tips.append(f"CPU: {cpu_tip}")
            if not tips:
                tips.append("Sistem durumu: ölçüm yokken '—' gösterilir")
            self.system_label.setToolTip(" | ".join(tips))
        except Exception as exc:
            logger.error("Sistem satırı çizilemedi: %s", exc)

    def _refresh_system_status(self) -> None:
        """core.system_monitor.SystemMonitor'dan gerçek ölçümleri okur.

        GPU/CPU: NVML/psutil ölçümü varsa yüzde, yoksa '—' (0 uydurulmaz).
        FPS: yalnızca capture örneği bağlıysa ölçülür, yoksa '—'.
        """
        try:
            from core.system_monitor import SystemMonitor
        except Exception as exc:
            logger.debug("SystemMonitor içe aktarılamadı: %s", exc)
            return
        try:
            snap = SystemMonitor.get_snapshot(self._system_capture)
        except Exception as exc:
            logger.debug("Sistem ölçümü okunamadı: %s", exc)
            return
        try:
            gpu = snap.get("gpu", {}) if isinstance(snap, dict) else {}
            cpu = snap.get("cpu", {}) if isinstance(snap, dict) else {}
            cap = snap.get("capture", {}) if isinstance(snap, dict) else {}
            fps_val = None
            if self._system_capture is not None:
                raw_fps = cap.get("fps")
                if raw_fps is not None:
                    fps_val = _safe_float(raw_fps)
                    if fps_val <= 0:
                        fps_val = None
            self.update_system_status(
                fps=fps_val,
                gpu_pct=(gpu.get("pct") if gpu.get("available") else None),
                cpu_pct=(cpu.get("pct") if cpu.get("available") else None),
                gpu_name=gpu.get("name"),
                gpu_temp_c=(gpu.get("temp") if gpu.get("available") else None),
                gpu_mem=(
                    f"{gpu.get('mem_used_gb', 0.0):.1f}/{gpu.get('mem_total_gb', 0.0):.1f} GB"
                    if gpu.get("available")
                    else None
                ),
                cpu_name=cpu.get("name"),
                cpu_freq_ghz=cpu.get("freq_ghz"),
            )
        except Exception as exc:
            logger.debug("Sistem ölçümü işlenemedi: %s", exc)

    def _sync_mode_buttons(self) -> None:
        """Prototip mod vurgusu: aktif mod düğmesi renkli çerçeveyle işaretlenir."""
        try:
            key = getattr(self, "_current_mode_key", "")
            gather_btn = self.__dict__.get("btn_gather")
            combat_btn = self.__dict__.get("btn_combat")
            if gather_btn is not None:
                if key == "gather":
                    gather_btn.setStyleSheet(
                        "QPushButton { border: 2px solid #43d68a;"
                        " background-color: rgba(67, 214, 138, 0.18); }"
                    )
                else:
                    gather_btn.setStyleSheet("")
            if combat_btn is not None:
                if key == "combat":
                    combat_btn.setStyleSheet(
                        "QPushButton { border: 2px solid #ff6a5e;"
                        " background-color: rgba(255, 106, 94, 0.18); }"
                    )
                else:
                    combat_btn.setStyleSheet("")
        except Exception as exc:
            logger.debug("Mod düğmesi vurgusu güncellenemedi: %s", exc)

    def _sync_control_buttons(self) -> None:
        """Prototip düğme durumları: çalışırken Başlat kapalı, Durdur açık;
        paused iken Durdur kapalı, Başlat/Devam açık. Oto-Kayıt WAITING
        değil WALKING sayılır ama paused bayrağı önceliklidir (High #7)."""
        try:
            key = getattr(self, "_current_state_key", "WAITING")
            paused = key in ("PAUSED", "DURAKLATILDI") or bool(
                getattr(self, "_paused_flag", False)
            )
            running = (not paused) and key in (
                "WALKING", "GATHERING", "FIGHTING", "COMBAT", "RUNNING",
                "YÜRÜYOR", "TOPLANIYOR", "SAVAŞTA", "ÇALIŞIYOR",
                # Player-driven mapping modes idle as WAITING while still live.
                "WAITING", "BEKLEMEDE", "READY", "IDLE",
            )
            resting = (not paused) and key in (
                "BREAK", "RESTING", "RESTING_BREAK", "MOLA", "DİNLENİYOR",
            )
            # WAITING only counts as "live" once the bot has been started at
            # least once (avoids enabling Durdur on cold boot).
            if key in (
                "WALKING", "GATHERING", "FIGHTING", "COMBAT", "RUNNING",
                "YÜRÜYOR", "TOPLANIYOR", "SAVAŞTA", "ÇALIŞIYOR",
            ) or resting:
                self._ever_started = True
            live = bool((running or resting) and self._ever_started)
            if hasattr(self, "btn_run"):
                self.btn_run.setEnabled(not live)
            if hasattr(self, "btn_stop"):
                self.btn_stop.setEnabled(bool(live))
            if hasattr(self, "btn_resume"):
                self.btn_resume.setEnabled(bool(paused and self._ever_started))
        except Exception as exc:
            logger.debug("Kontrol düğmeleri senkronize edilemedi: %s", exc)

    @staticmethod
    def _format_gathering_title_with_stage(raw_title: str, stage: int, max_stages: int) -> str:
        """
        Herhangi bir toplama başlığının Türkçe durum ve [aşama/maks] rozetini içermesini sağlar:
        ör. 'Durum: Toplanıyor [2/3] — Gathering node #12 [2/3]' veya 'Durum: Toplanıyor [2/3]'
        """
        cleaned = re.sub(r"\s*\[\d+\s*/\s*\d+\]\s*$", "", (raw_title or "").strip())
        # Aşama 0 iken '1/3' yazılmaz: gerçek aşama 0/3'tür.
        display_stage = max(0, min(stage, max(1, max_stages)))
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
            # K-1: gerçek aşama geldi — rozet artık "—/3" değil, sayı gösterir.
            self._harvest_stage_known = True
            self._refresh_harvest_badge_style()

            # Başlık yalnızca gerçekten toplanıyorsa güncellenir; toplanmıyor
            # iken aşama > 0 diye başlığı 'Toplanıyor' yapmak uydurmadır.
            current_title = self.title_label.text()
            is_gathering = (
                self._current_state in ("GATHERING", "TOPLANIYOR")
                or "toplanıyor" in current_title.lower()
                or "gathering" in current_title.lower()
            )
            if is_gathering and self._current_harvest_stage > 0:
                updated_title = self._format_gathering_title_with_stage(
                    current_title,
                    self._current_harvest_stage,
                    self._max_harvest_stages,
                )
                self.title_label.setText(updated_title)

            self.harvest_stage_updated.emit(self._current_harvest_stage, self._max_harvest_stages)
        except Exception as exc:
            logger.error("update_harvest_stage(%s, %s) hatası: %s", current_stage, max_stages, exc)

    @staticmethod
    def _format_last_mapped(value: Any) -> str:
        """Y-1: last_mapped değerini (dict/str) kısa HUD dizesine çevirir.

        Kartograf biçimi: {"type": "node", "node_id": 2, "map_x": 710,
        "map_y": 560, "timestamp": ..., "confidence": 0.93}. Bilinmeyen
        biçim olduğu gibi yazılır; boş/None "—" olur (uydurma yok).
        """
        if value is None or value == "":
            return "—"
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            kind = str(value.get("type") or "")
            nid = value.get("node_id", value.get("waypoint_id"))
            label = ""
            if nid is not None:
                label = f"Node {nid}" if kind in ("node", "") else f"{kind} {nid}"
            elif kind:
                label = kind
            if not label:
                label = str(value.get("resource_type") or value.get("name") or "—")
            return label
        return str(value)

    def set_panic_lock(self, panic_lock: Any) -> None:
        """A grubu devri: canlı PanicAlertSystem kaynağını bağlar.

        Kilit get_panic_status() sunuyorsa 1 sn'lik HUD zamanlayıcısı
        durumu yoklar ve şeridi günceller. Kilit yoksa (None) disk
        yedeği (refresh_panic_from_disk) kullanılır. Asla exception
        fırlatmaz; uyumsuz nesne yoksayılır.
        """
        try:
            if panic_lock is not None and not callable(getattr(panic_lock, "get_panic_status", None)):
                logger.debug("set_panic_lock: nesnede get_panic_status yok, yoksayıldı")
                return
            self._panic_lock = panic_lock
        except Exception as exc:
            logger.debug("set_panic_lock hatası: %s", exc)

    def refresh_panic_from_disk(self) -> Optional[Dict[str, Any]]:
        """A grubu devri: logs/alarms/ altındaki son panik kaydını okur.

        Okuyucu A dalında eklendi (PanicAlertSystem.load_panic_history_from_disk);
        bu dalda yoksa getattr ile None döner — çökme yok. Kayıt varsa
        son panik şeritte nötr gri gösterilir ("son panik (disk)").
        Canlı kilit aktifken disk kaydı canlıyı ezmez.
        """
        try:
            from core.opsec_crypto import PanicAlertSystem

            reader = getattr(PanicAlertSystem, "load_panic_history_from_disk", None)
            if not callable(reader):
                return None
            records = reader(limit=1)
            if not records:
                return None
            last = dict(records[-1])
            try:
                current = self._panic_lock.get_panic_status() if self._panic_lock is not None else None
            except Exception:
                current = None
            if isinstance(current, dict) and current.get("panic_triggered"):
                return last
            self.update_panic_status(
                {
                    "panic_triggered": False,
                    "reason": last.get("reason"),
                    "details": last.get("details"),
                    "timestamp": last.get("timestamp"),
                    "elapsed_sec": None,
                    "from_disk": True,
                }
            )
            return last
        except Exception as exc:
            logger.debug("Disk panik geçmişi okunamadı: %s", exc)
            return None

    def update_panic_status(self, status: Optional[Dict[str, Any]]) -> None:
        """A grubu devri: panik şeridini günceller.

        status = PanicAlertSystem.get_panic_status() biçimi
        ({panic_triggered, inputs_halted, reason, details, timestamp,
        elapsed_sec, history_count}) veya refresh_panic_from_disk'in
        ürettiği from_disk'li sözlük. Aynı durum tekrar çizilmez/loglanmaz.
        Kilit yokken şerit gizlenir; diskten gelen son panik nötr gösterilir.
        fix/v20-panic-combat: dedup anahtarı artık (triggered + reason
        + details) — eski anahtar (triggered + from_disk + reason)
        canlı/disk kaynakları arasında gidip gelen çağrılarda her
        seferinde değişiyor, şerit her yenilemede yeniden açılıp
        "HUD panik şeridi açıldı" logunu her saniye tekrarlıyordu
        (canlı v19: 1 sn, 2 sn, ... 30 sn). Anahtara elapsed_sec ve
        timestamp GİRMEZ: ikisi de epizot boyunca (veya üreticiden
        üreticiye) değişir ve dedup'u bozar. details epizot başına
        sabittir ve tetiklenme anındaki süreyi taşır ("... (123.2s)"),
        o yüzden yeni epizot (yeni details) şeridi yeniden açar.
        Aynı epizot tekrar geldiğinde yalnızca şeritteki geçen-süre
        metni SESSİZCE tazelenir (log yok, sinyal yok).
        """
        try:
            if not isinstance(status, dict):
                return
            triggered = bool(status.get("panic_triggered"))
            reason = str(status.get("reason") or "")
            details = str(status.get("details") or "")
            elapsed = status.get("elapsed_sec")
            from_disk = bool(status.get("from_disk"))
            # Dedup anahtarı: aynı kilit hangi kaynaktan gelirse gelsin
            # aynı anahtardır; yeni epizot (yeni details) yeniden açar.
            # elapsed_sec / timestamp bilerek dışarıda bırakıldı.
            signature = "|".join(
                [
                    "1" if triggered else "0",
                    reason,
                    details,
                ]
            )
            if signature == self.__dict__.get("_panic_signature"):
                # Aynı epizot: şerit zaten açık — yalnızca geçen-süre
                # sayacını sessizce tazele (log/sinyal yok).
                if triggered:
                    label = getattr(self, "panic_label", None) or getattr(
                        self, "_panic_label", None
                    )
                    if label is not None:
                        try:
                            elapsed_txt = (
                                f" ({elapsed:.0f} sn)"
                                if isinstance(elapsed, (int, float))
                                else ""
                            )
                            unlock = ""
                            if str(reason or "") == "PROMPT_STUCK_10S":
                                unlock = " · Açmak: Başlat/Devam veya Ctrl+Alt+F10"
                            text = (
                                f"⛔ PANİK KİLİT: {reason or 'sebep bilinmiyor'}"
                                f"{elapsed_txt}{unlock}"
                            )
                            if details:
                                text += f" — {details}"
                            label.setText(text)
                        except Exception:
                            pass
                return
            self._panic_signature = signature
            label = getattr(self, "panic_label", None) or getattr(self, "_panic_label", None)
            if label is None:
                return
            if triggered:
                elapsed_txt = f" ({elapsed:.0f} sn)" if isinstance(elapsed, (int, float)) else ""
                unlock = ""
                soft = bool(status.get("soft_pause")) if isinstance(status, dict) else False
                if soft or str(reason or "") == "PROMPT_STUCK_10S":
                    unlock = " · Açmak: Başlat/Devam veya Ctrl+Alt+F10"
                    if soft:
                        text = (
                            f"⚠ Takılı istem (PROMPT_STUCK): bot duraklatıldı"
                            f"{elapsed_txt}{unlock}"
                        )
                        if details:
                            text += f" — {details}"
                        label.setText(text)
                        label.setStyleSheet(
                            """
                            QLabel {
                                background-color: rgba(255, 193, 7, 0.14);
                                color: #e7c45a;
                                border: 1px solid rgba(255, 193, 7, 0.40);
                                border-radius: 6px;
                                font-family: 'Barlow', 'Segoe UI', sans-serif;
                                font-size: 12px;
                                font-weight: 700;
                                padding: 3px 8px;
                            }
                            """
                        )
                        label.setVisible(True)
                        logger.warning("HUD prompt-stuck soft şeridi: %s", text)
                        try:
                            self.panic_status_updated.emit(dict(status))
                        except Exception:
                            pass
                        return
                text = f"⛔ PANİK KİLİT: {reason or 'sebep bilinmiyor'}{elapsed_txt}{unlock}"
                if details:
                    text += f" — {details}"
                label.setText(text)
                label.setStyleSheet(
                    """
                    QLabel {
                        background-color: rgba(255, 106, 94, 0.16);
                        color: #ff8d84;
                        border: 1px solid rgba(255, 106, 94, 0.45);
                        border-radius: 6px;
                        font-family: 'Barlow', 'Segoe UI', sans-serif;
                        font-size: 12px;
                        font-weight: 700;
                        padding: 3px 8px;
                    }
                    """
                )
                label.setVisible(True)
                logger.warning("HUD panik şeridi açıldı: %s", text)
            elif from_disk and reason:
                label.setText(f"Son panik (disk kaydı): {reason}")
                label.setStyleSheet(
                    """
                    QLabel {
                        background-color: rgba(255, 255, 255, 0.05);
                        color: #9fb2c3;
                        border: 1px solid rgba(255, 255, 255, 0.12);
                        border-radius: 6px;
                        font-family: 'Barlow', 'Segoe UI', sans-serif;
                        font-size: 11px;
                        font-weight: 600;
                        padding: 3px 8px;
                    }
                    """
                )
                label.setVisible(True)
            else:
                label.setVisible(False)
            try:
                self.panic_status_updated.emit(dict(status))
            except Exception:
                pass
        except Exception as exc:
            logger.debug("update_panic_status hatası: %s", exc)

    @pyqtSlot(int)
    def update_minimap_mapped(self, count: int) -> None:
        """Sağ üst Minimap'ten otomatik kaydedilen canlı nokta sayısını günceller."""
        try:
            self._minimap_mapped_count = max(0, int(count))
            # Sayı etikette görünür; yalnızca tooltip'te değil.
            self.minimap_mapped_label.setText(f"Harita: {self._minimap_mapped_count}")
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
        hp_pct: Optional[float] = None,
        mp_pct: Optional[float] = None,
        active_scrolls: Any = 0,
        selective_target: str = "",
        active_buffs: Optional[List[str]] = None,
        selected_material: str = "",
    ) -> dict:
        """
        Aktif tüketilebilir/iksir/parşömen durumunu (hp_potion, mp_potion, buff_scroll kalan süre/hazır durumu)
        ve Seç-Topla hedef materyal rozetini günceller ve güncel durum sözlüğünü döndürür.
        NOT [O4, F-0592]: hp/mp varsayılanı None'dur — %100 uydurulmaz; None
        mevcut değeri korur (başlangıçta bilinmiyor). Canlı yol (update_state)
        ölçülmüş değeri geçirir.
        """
        try:
            # None = "veri yok, mevcut değeri koru": %100 varsayımı uydurma olur.
            # (İmza dondurmalı olduğundan parametre varsayılanı 100.0 kalır;
            # canlı yol update_state üzerinden her zaman ölçülmüş değer geçer.)
            if hp_pct is None:
                hp_val = self.__dict__.get("_hp_pct")
            else:
                hp_val = max(0.0, min(100.0, float(hp_pct)))
            if mp_pct is None:
                mp_val = self.__dict__.get("_mp_pct")
            else:
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

            if hp_val is None:
                hp_potion_state = "BİLİNMİYOR"
            else:
                # F-0540: rozet combat ile aynı eşiği kullanır (HP 65).
                hp_potion_state = "TRIGGERED" if hp_val < float(HP_POTION_THRESHOLD_PCT) else "READY"
            if mp_val is None:
                mp_potion_state = "BİLİNMİYOR"
            else:
                # F-0540: rozet combat ile aynı eşiği kullanır (MP 45).
                mp_potion_state = "TRIGGERED" if mp_val < float(MP_POTION_THRESHOLD_PCT) else "READY"
            buff_scroll_state = f"ACTIVE ({scrolls_val})" if scrolls_val > 0 else "READY"

            hp_txt = f"{int(round(hp_val))}%" if hp_val is not None else "?%"
            mp_txt = f"{int(round(mp_val))}%" if mp_val is not None else "?%"
            badge_text = f"HP {hp_txt} · MP {mp_txt} · S:{scrolls_val}"
            selective_badge_text = (
                f"Hedef: {target_val}" if target_val else "Seç-Topla: Tümü"
            )

            status_payload: Dict[str, Any] = {
                "hp_pct": round(hp_val, 1) if hp_val is not None else None,
                "mp_pct": round(mp_val, 1) if mp_val is not None else None,
                "active_scrolls": scrolls_val,
                "selective_target": target_val,
                "selective_active": bool(target_val),
                "hp_potion": hp_potion_state,
                "mp_potion": mp_potion_state,
                "buff_scroll": buff_scroll_state,
                "consumables_ready": hp_potion_state == "READY" and mp_potion_state == "READY",
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
                    if hp_val is not None and mp_val is not None
                    else f"HP İksiri: {hp_potion_state} | MP İksiri: {mp_potion_state} | Parşömen: {buff_scroll_state}"
                )
                # Rozet ilk gerçek güncellemede görünür olur (başlangıçta gizli).
                c_lbl.setVisible(True)
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
                        font-size: 11px;
                        font-weight: 700;
                        letter-spacing: 0.04em;
                        padding: 3px 8px;
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

    def set_selective_gather_status(
        self, enabled: bool, materials: Optional[List[str]] = None
    ) -> dict:
        """Seç-Topla durumunu HUD'da gösterir (main.py burayı çağırır; daha
        önce bu yöntem yoktu, çağrı sessizce boşa düşüyordu).

        enabled=False veya boş liste -> "Seç-Topla: Tümü" (filtre kapalı).
        Aksi halde hedef materyaller rozete yazılır ve rozet görünür olur.
        Güncel durum sözlüğünü döndürür.
        """
        try:
            mats = [str(m).strip() for m in (materials or []) if str(m).strip()]
            self._selective_enabled = bool(enabled)
            self._selective_materials = mats if enabled else []
            target_val = ", ".join(mats) if (enabled and mats) else ""
            self._selective_target = target_val
            status = self.update_consumable_and_selective_status(
                hp_pct=None,
                mp_pct=None,
                active_scrolls=self.__dict__.get("_active_scrolls", 0),
                selective_target=target_val,
            )
            status["selective_enabled"] = bool(enabled)
            status["selected_materials"] = list(self._selective_materials)
            s_lbl = self.__dict__.get("selective_badge_label")
            if s_lbl is not None:
                if enabled and mats:
                    s_lbl.setText(f"Seç-Topla: {', '.join(mats)}")
                    s_lbl.setToolTip(f"Seç-Topla aktif hedefler: {', '.join(mats)}")
                    s_lbl.setVisible(True)
                elif enabled:
                    s_lbl.setText("Seç-Topla: Tümü")
                    s_lbl.setToolTip("Seç-Topla açık, hedef filtre yok (tüm materyaller)")
                    s_lbl.setVisible(True)
                else:
                    s_lbl.setText("Seç-Topla: Tümü")
                    s_lbl.setToolTip("Seç-Topla: Tüm Materyaller")
                    s_lbl.setVisible(False)
            return dict(status)
        except Exception as exc:
            logger.error("set_selective_gather_status hatası: %s", exc)
            return dict(self.__dict__.get("_consumable_status", {}))

    def get_overlay_telemetry_snapshot(self) -> dict:
        """
        Mevcut durum, mod, 3x hasat aşaması (1/3..3/3), minimap sayacı ve tüketilebilir
        durumunu tek sözlükte döndürür.
        """
        try:
            stage = max(0, int(self.__dict__.get("_current_harvest_stage", 0)))
            max_s = max(1, int(self.__dict__.get("_max_harvest_stages", 3)))
            # Aşama 0 iken '1/3' yazılmaz.
            harvest_phase_str = f"{min(stage, max_s)}/{max_s}"
            harvest_raw_str = f"{min(stage, max_s)}/{max_s}"
            consumables = dict(self.__dict__.get("_consumable_status", {}))
            hp_raw = self.__dict__.get("_hp_pct")
            mp_raw = self.__dict__.get("_mp_pct")

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
                "hp_pct": float(hp_raw) if hp_raw is not None else None,
                "mp_pct": float(mp_raw) if mp_raw is not None else None,
                "active_scrolls": int(self.__dict__.get("_active_scrolls", 0)),
                "selective_target": str(self.__dict__.get("_selective_target", "")),
                "hp_potion": consumables.get("hp_potion", "BİLİNMİYOR"),
                "mp_potion": consumables.get("mp_potion", "BİLİNMİYOR"),
                "buff_scroll": consumables.get("buff_scroll", "BİLİNMİYOR"),
                "consumables": consumables,
                "consumable_status": consumables,
                # is_minimized bir property'dir (__dict__'te yoktur); gerçek
                # alan _minimized'dir — eskisi hep False döndürüyordu.
                "is_minimized": bool(self.__dict__.get("_minimized", False)),
                "click_through": bool(self.__dict__.get("_click_through_enabled", False)),
                # Prototip satırları (canlı kaynaklı; bilinmeyen None/"—").
                "state_key": self.__dict__.get("_current_state_key", "WAITING"),
                "mode_key": self.__dict__.get("_current_mode_key", ""),
                "route_current": self.__dict__.get("_route_current"),
                "route_total": self.__dict__.get("_route_total"),
                "route_loops": self.__dict__.get("_route_loops"),
                "route_zone": self.__dict__.get("_route_zone", ""),
                "target_node_id": self.__dict__.get("_target_node_id"),
                "target_distance_m": self.__dict__.get("_target_distance_m"),
                "target_material": self.__dict__.get("_target_material", ""),
                "system_status": dict(self.__dict__.get("_system_status", {})),
                "detection_boxes_visible": bool(
                    self.__dict__.get("_detection_boxes_visible", True)
                ),
                "minimap_visible": bool(self.__dict__.get("_minimap_visible", True)),
            }
        except Exception as exc:
            logger.error("get_overlay_telemetry_snapshot hatası: %s", exc)
            return {
                "state": self.__dict__.get("_current_state", "WAITING"),
                "mode": self.__dict__.get("_current_mode", "ROTA MODU"),
                "harvest_stage": 0,
                "max_harvest_stages": 3,
                "harvest_phase": "0/3",
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
    def _on_stop_clicked(self) -> None:
        """Durdur: pause only. Never toggles into resume."""
        try:
            self.stop_clicked.emit()
        except Exception as exc:
            logger.error("stop_clicked sinyali gönderilemedi: %s", exc)

    @pyqtSlot()
    def _on_settings_clicked(self) -> None:
        """Overlay dişli: Ayarlar aç; bot pause/resume'a dokunma (High #5)."""
        try:
            self.settings_clicked.emit()
        except Exception as exc:
            logger.error("settings_clicked sinyali gönderilemedi: %s", exc)

    @pyqtSlot()
    def _on_quit_clicked(self) -> None:
        try:
            self.quit_requested.emit()
        except Exception as exc:
            logger.error("quit_requested sinyali gönderilemedi: %s", exc)

    def bind_f9_menu(self, menu: Any) -> None:
        """HUD düğmeleri mevcut F9 menüsünün eylemlerine bağlanır."""
        self._f9_menu = menu

    def bind_live_sources(
        self, route_manager: Any = None, worker: Any = None, capture: Any = None
    ) -> None:
        """Canlı kaynak referanslarını tutar ve türetilen satırları tazeler.

        route_manager: aktif bölge/rota + düğüm listesi (rota ilerlemesi, hedef).
        worker: active_node / active_node_distance_m (hedef, rota sırası).
        capture: ekran yakalama örneği (FPS ölçümü; yoksa FPS '—' kalır).
        main.py bu yöntemi route_manager + worker ile çağırır (capture
        sonradan bind_system_monitor ile de verilebilir).
        """
        if route_manager is not None:
            self._route_manager = route_manager
        if worker is not None:
            self._worker = worker
        if capture is not None:
            self._system_capture = capture
        try:
            self._refresh_live_derived()
        except Exception as exc:
            logger.debug("Canlı kaynak ilk tazeleme hatası: %s", exc)
        try:
            self._refresh_system_status()
        except Exception as exc:
            logger.debug("Sistem durumu ilk okuma hatası: %s", exc)

    def set_hotkey_labels(self, node: str, menu: str, pause: str) -> None:
        """Kısayol ipuçlarını günceller; yatay panel düzeni aynı kalır."""
        try:
            if hasattr(self, "f8_shortcut"):
                self.f8_shortcut.setToolTip(f"{node}: Nokta kaydet")
            if hasattr(self, "f9_shortcut"):
                self.f9_shortcut.setToolTip(f"{menu}: Menü")
            if hasattr(self, "f10_shortcut"):
                self.f10_shortcut.setToolTip(f"{pause}: çalışırken Durdur, duraklatılınca Başlat")
            if hasattr(self, "btn_node"):
                self.btn_node.setToolTip(f"{node}: Nokta kaydet")
            if hasattr(self, "btn_menu"):
                self.btn_menu.setToolTip(f"{menu}: Menü (duraklatmaz / devam ettirmez)")
            if hasattr(self, "btn_run"):
                self.btn_run.setToolTip(f"{pause}: Başlat / devam")
            if hasattr(self, "btn_stop"):
                self.btn_stop.setToolTip(f"{pause}: yalnızca Durdur (devam etmez)")
        except Exception as exc:
            logger.error("HUD kısayol etiketleri güncellenemedi: %s", exc)

    def _dispatch_f9_action(self, action_id: str) -> None:
        """Devam Et, Savaş ve Toplama düğmeleri F9 menüsündeki aynı eylemi çalıştırır."""
        try:
            menu = getattr(self, "_f9_menu", None)
            if menu is not None and hasattr(menu, "_trigger_action"):
                menu._trigger_action(action_id)
                return
            # Menü bağlı değilse sessizce yutulmaz: uyarı + F9 sinyaliyle
            # ana uygulamaya devredilir (menüyü açması için).
            logger.warning(
                "HUD F9 eylemi (%s) için menü bağlı değil; f9_clicked sinyali gönderiliyor",
                action_id,
            )
            self.f9_clicked.emit()
        except Exception as exc:
            logger.error("HUD F9 eylemi çalışmadı (%s): %s", action_id, exc)

    def nativeEvent(self, eventType: Any, message: Any) -> Tuple[bool, int]:
        """Tıklama gelsin ama pencere öne çıkıp oyunun odağını almasın."""
        try:
            if sys.platform == "win32" and eventType == b"windows_generic_MSG":
                msg = ctypes.wintypes.MSG.from_address(int(message))
                if int(msg.message) == 0x0021:  # WM_MOUSEACTIVATE
                    return True, 3  # MA_NOACTIVATE
                if int(msg.message) == 0x0084:  # WM_NCHITTEST — seçici tıklama geçişi
                    if getattr(self, "_click_through_enabled", False):
                        try:
                            l_param = int(msg.lParam)
                            x = l_param & 0xFFFF
                            y = (l_param >> 16) & 0xFFFF
                            if x >= 0x8000:
                                x -= 0x10000
                            if y >= 0x8000:
                                y -= 0x10000
                            local = self.mapFromGlobal(QPoint(x, y))
                        except Exception:
                            local = self.mapFromGlobal(QCursor.pos())
                        widget = self.childAt(local)
                        while widget is not None and widget is not self:
                            if isinstance(widget, (QPushButton, ClickableShortcutWidget)):
                                return True, 1  # HTCLIENT — düğme tıklanabilir
                            widget = widget.parentWidget()
                        return True, -1  # HTTRANSPARENT — boş alan oyuna geçer
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
            # Yavaş canlı tazeleme (5 sn'de bir): sistem ölçümü + rota/hedef.
            # Ölçüm yoksa satırlar '—' kalır; hiçbir değer uydurulmaz.
            self._slow_tick += 1
            if self._slow_tick % 5 == 0:
                try:
                    self._refresh_system_status()
                except Exception:
                    pass
                try:
                    self._refresh_live_derived()
                except Exception:
                    pass
                # A grubu devri: canlı panik kilidini yokla (5 sn'de bir;
                # update_panic_status aynı durumu tekrar çizmez/loglamaz).
                try:
                    lock = self.__dict__.get("_panic_lock")
                    if lock is not None:
                        status = lock.get_panic_status()
                        if isinstance(status, dict):
                            self.update_panic_status(status)
                except Exception:
                    pass
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
            theme = self.STATE_THEMES.get(normalized)
            if theme is None:
                # Bilinmeyen durum ham haliyle rozete yazılmaz; güvenli
                # varsayılan BEKLEMEDE + uyarı.
                logger.warning("OverlayHUD: bilinmeyen durum '%s', BEKLEMEDE sayıldı", state_key)
                normalized = "BEKLEMEDE"
                theme = self.STATE_THEMES["BEKLEMEDE"]
            badge_text, accent_hex, badge_bg, badge_fg = theme
            self._current_state = badge_text
            self._current_state_key = normalized
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
                    font-size: 12px;
                    font-weight: 700;
                    letter-spacing: 0.06em;
                    padding: 1px 7px;
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
        """İkincil mod kapsül rozetini güncelle.

        State machine'in gerçek rozet adları tanınır: 'TOPLAMA (ROUTE)',
        'SAVAŞ (COMBAT)' (core/state_machine.py MODE_BADGE_MAP), 'MANUAL
        RECORD', 'AUTO RECORD', 'AUTO CARTOGRAPHY' + 'route'/'combat'/
        'gather' kısaltmaları. Yalnızca gerçekten bilinmeyen ad uyarı
        verip yoksayılır; önceki mod korunur.
        """
        try:
            raw_mode = (mode or "ROUTE").strip()
            resolved = self._resolve_mode_label(raw_mode)
            if resolved is None:
                # Bilinmeyen mod ham haliyle rozete yazılmaz; önceki mod korunur.
                logger.warning("OverlayHUD: bilinmeyen mod '%s', yoksayıldı", mode)
                return
            self._current_mode = resolved
            self._current_mode_key = self._mode_key_for(raw_mode, resolved)
            self.mode_label.setText(resolved)
            self._sync_mode_buttons()
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
        - stat2_title / col2_title, stat2_val / col2_val / region_route / region / route / ready_resting / f_pressed
        - stat3_title / col3_title, stat3_val / col3_val / failed_in_row / match_pct / last_saved / last_mapped
        - panic_status: {panic_triggered, reason, details, elapsed_sec} — panik kilit şeridi (A grubu devri)
        - hp_pct / mp_pct / active_scrolls / selective_target: canlı vital + tüketilebilir rozeti
        - selective_gather_enabled / selected_materials: Seç-Topla rozeti
        - radar: iç içe radar sözlüğü {blips, target_bearing_deg, target_distance_m,
          target_name, player_pos, heading_deg, range_m}
        - version: sürüm etiketi
        """
        if not isinstance(data, dict):
            logger.warning("OverlayHUD.update_state dict bekliyordu, alınan: %s", type(data).__name__)
            return
        if self.__dict__.get("_ui_ready") is False:
            # _init_ui başarısız olduysa yarım HUD güncellenmez.
            logger.warning("OverlayHUD arayüzü kurulamadı, update_state yoksayıldı")
            return

        try:
            # 1. Durum Rozeti & Sol Vurgu Kenarlığı
            raw_state = data.get("state", data.get("badge", self._current_state or "WAITING"))
            state_val = str(raw_state).upper()
            if "paused" in data:
                self._paused_flag = bool(data.get("paused"))
            elif state_val in ("PAUSED", "DURAKLATILDI"):
                self._paused_flag = True
            elif state_val in (
                "WALKING", "GATHERING", "FIGHTING", "COMBAT", "RUNNING",
                "YÜRÜYOR", "TOPLANIYOR", "SAVAŞTA", "ÇALIŞIYOR",
            ):
                self._paused_flag = False
            self._apply_state_theme(state_val)
            if state_val in (
                "WALKING", "GATHERING", "FIGHTING", "COMBAT", "RUNNING",
                "YÜRÜYOR", "TOPLANIYOR", "SAVAŞTA", "ÇALIŞIYOR",
            ):
                self._ever_started = True

            # 2. Mod Kapsül Rozeti
            raw_mode = data.get("mode", data.get("mode_badge", None))
            if raw_mode is not None:
                self.set_mode(str(raw_mode))

            # 3. Düğüm Başına 3x Toplama Aşaması Güncellemesi
            # Aşama yalnızca açıkça verildiğinde değişir: TOPLANIYOR diye
            # 1'e uydurulmaz, başka duruma geçildi diye 0'a sıfırlanmaz.
            stage_val = data.get("harvest_stage", data.get("current_stage", None))
            max_stage_val = data.get("max_harvest_stages", data.get("max_stages", self._max_harvest_stages))
            if stage_val is not None:
                self._current_harvest_stage = max(0, int(stage_val))
                self._max_harvest_stages = max(1, int(max_stage_val))
                # K-1: gerçek aşama geldi — rozet artık "—/3" değil.
                self._harvest_stage_known = True
            self._refresh_harvest_badge_style()

            # 3b. Panik kilit durumu (A grubu devri): payload'da
            # "panic_status" sözlüğü varsa şeridi güncelle.
            panic_val = data.get("panic_status", None)
            if isinstance(panic_val, dict):
                self.update_panic_status(panic_val)

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

            # 4b. Canlı Vital (HP/MP) + Tüketilebilir Rozeti — üretim yolu:
            # işçi HUD verisindeki hp_pct/mp_pct anahtarları rozeti besler.
            vitals_keys = ("hp_pct", "mp_pct", "active_scrolls", "selective_target", "selected_material")
            if any(k in data for k in vitals_keys):
                self.update_consumable_and_selective_status(
                    hp_pct=data.get("hp_pct"),
                    mp_pct=data.get("mp_pct"),
                    active_scrolls=data.get("active_scrolls", self.__dict__.get("_active_scrolls", 0)),
                    selective_target=str(
                        data.get(
                            "selective_target",
                            data.get("selected_material", self.__dict__.get("_selective_target", "")),
                        )
                    ),
                )

            # 4d. Prototip satırları: hedef / rota ilerlemesi / sistem / tespitler.
            # Açık anahtarlar doğrudan yazılır; yoksa bağlı canlı kaynaklardan
            # türetilir (update_state sonundaki _refresh_live_derived);
            # hiçbir alan uydurulmaz, bilinmeyen "—" kalır.
            target_keys = ("target_node_id", "target_id", "node_id", "target", "target_name")
            if any(k in data for k in target_keys) or "target_distance_m" in data or "target_material" in data:
                mat = data.get("target_material", data.get("target_resource", None))
                if mat is None and isinstance(data.get("target"), str):
                    mat = data.get("target")
                nid = data.get("target_node_id", data.get("target_id", data.get("node_id", None)))
                if nid is None and isinstance(data.get("target"), (int, dict)):
                    nid = data.get("target")
                tname = data.get("target_name", None)
                if mat is None and isinstance(tname, str) and tname:
                    mat = tname
                self.update_target(
                    node_id=nid,
                    distance_m=data.get("target_distance_m", data.get("target_distance", None)),
                    material=mat,
                )
            route_keys = ("route_current", "route_total", "route_loops", "route_progress")
            if any(k in data for k in route_keys) or "zone" in data or "region" in data:
                cur = data.get("route_current", data.get("current_node", None))
                tot = data.get("route_total", data.get("total_nodes", None))
                if cur is None and isinstance(data.get("route_progress"), (int, float)):
                    prog = _safe_float(data.get("route_progress"), -1.0)
                    if 0.0 <= prog <= 1.0 and tot is not None:
                        cur = int(round(prog * int(tot)))
                self.update_route_progress(
                    current=(int(cur) if cur is not None else None),
                    total=(int(tot) if tot is not None else None),
                    loops=(int(data["route_loops"]) if data.get("route_loops") is not None else None),
                    zone=data.get("zone", data.get("region", None)),
                )
            if isinstance(data.get("system"), dict):
                sys_keys = (
                    "fps", "gpu_pct", "cpu_pct", "gpu_name", "gpu_temp_c",
                    "gpu_mem", "cpu_name", "cpu_freq_ghz",
                )
                sys_kwargs = {k: v for k, v in data["system"].items() if k in sys_keys}
                if sys_kwargs:
                    self.update_system_status(**sys_kwargs)
            detections = data.get("detections", data.get("radar_blips", None))
            if isinstance(detections, list):
                try:
                    self.set_radar_blips(list(detections))
                except Exception:
                    pass

            # 4c. Seç-Topla durumu (main.py handle_selective_gather_changed
            # buraya update_state ile de uğrar).
            if "selective_gather_enabled" in data or "selected_materials" in data:
                mats = data.get("selected_materials", None)
                if isinstance(mats, str):
                    mats = [mats]
                self.set_selective_gather_status(
                    bool(data.get("selective_gather_enabled", True)),
                    list(mats) if isinstance(mats, (list, tuple)) else None,
                )

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
                    stage_for_title = self._current_harvest_stage
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
                stage_for_title = self._current_harvest_stage
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
                self.stat1_title.setText(self._translate_column(str(col1_t)))
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
                self.stat2_title.setText(self._translate_column(str(col2_t)))
            elif "region_route" in data and data["region_route"]:
                self.stat2_title.setText(self._translate_column(str(data["region_route"])))
            elif "region" in data or "route" in data:
                self.stat2_title.setText("BÖLGE · ROTA")
            elif self._current_mode == "MANUEL KAYIT" and ("ready_count" in data or "nodes_summary" in data):
                self.stat2_title.setText("NOKTALAR (AKTİF / DİNLENEN)")
            elif "ready_count" in data or "nodes_summary" in data:
                if self.stat2_title.text() in ("F BASILMA", "F PRESSED", "NODES", "NOKTALAR", ""):
                    self.stat2_title.setText("BÖLGE · ROTA")
            elif "f_pressed" in data and "ready_count" not in data and "nodes_summary" not in data:
                self.stat2_title.setText("F BASILMA")

            col2_v = data.get("stat2_val", data.get("col2_val", None))
            if col2_v is not None:
                # P2-7 (K-02/K-06): düğüm durumu sözcükleri Türkçe'ye
                # çevrilir — P1-6 `effective_node_state` sözlüğündeki
                # "failed"/"quarantined" eskiden İngilizce kalıyordu
                # (ready/resting çevrilirken).
                val_str = (
                    str(col2_v)
                    .replace("ready", "aktif")
                    .replace("resting", "dinlenen")
                    .replace("failed", "başarısız")
                    .replace("quarantined", "karantinada")
                )
                self.stat2_value.setText(val_str)
            elif "nodes_summary" in data:
                val_str = (
                    str(data["nodes_summary"])
                    .replace("ready", "aktif")
                    .replace("resting", "dinlenen")
                    .replace("failed", "başarısız")
                    .replace("quarantined", "karantinada")
                )
                self.stat2_value.setText(val_str)
            elif "ready_count" in data or "resting_count" in data:
                ready_c = int(data.get("ready_count", 0))
                resting_c = int(data.get("resting_count", 0))
                self.stat2_value.setText(f"{ready_c} aktif · {resting_c} dinlenen")
            elif "region_route" in data and data["region_route"]:
                self.stat2_value.setText(str(data["region_route"]))
            elif "region" in data or "route" in data:
                region = str(data.get("region", "")).strip()
                route = str(data.get("route", "")).strip()
                self.stat2_value.setText(" - ".join(p for p in (region, route) if p))
            elif "f_pressed" in data:
                self.stat2_value.setText(str(data["f_pressed"]))

            # Sütun 3: ARD ARDA BAŞARISIZ / SON KAYDEDİLEN / EŞLEŞME ORANI
            # Y-1: "last_mapped" (kartografın last_mapped_info'su) varsa
            # SON HARİTALANAN gösterilir — C grubunun rehidrasyon
            # düzeltmesi last_saved_str'i besler, bu anahtar kartograf
            # kaynağını doğrudan gösterir; ikisi de yoksa eski davranış.
            col3_t = data.get("stat3_title", data.get("col3_title", None))
            if col3_t is not None:
                self.stat3_title.setText(self._translate_column(str(col3_t)))
            elif "last_mapped" in data:
                self.stat3_title.setText("SON HARİTALANAN")
            elif "last_saved" in data or self._current_mode == "MANUEL KAYIT":
                self.stat3_title.setText("SON KAYDEDİLEN")
            elif "failed_in_row" in data or "failed_text" in data:
                self.stat3_title.setText("ARD ARDA BAŞARISIZ")
            elif "match_pct" in data and "failed_in_row" not in data:
                self.stat3_title.setText("EŞLEŞME ORANI")

            col3_v = data.get("stat3_val", data.get("col3_val", None))
            if col3_v is not None:
                self.stat3_value.setText(str(col3_v))
            elif "last_mapped" in data:
                self.stat3_value.setText(self._format_last_mapped(data["last_mapped"]))
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
                radar_payload = data.get("radar") if isinstance(data.get("radar"), dict) else {}
                radar_keys = (
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
                has_radar_data = any(k in data for k in radar_keys) or bool(radar_payload)
                if has_radar_data:
                    def _radar_get(key: str, default: Any = None) -> Any:
                        if key in data:
                            return data[key]
                        return radar_payload.get(key, default)

                    p_pos = _radar_get(
                        "player_pos",
                        (_radar_get("player_x", 0.0), _radar_get("player_y", 0.0)),
                    )
                    if isinstance(p_pos, dict):
                        # P3-8 (S-060): üretici dict gönderir
                        # {"map_x","map_y",...}; dict dizi değildir — harita
                        # koordinatları okunur, (0,0)'a düşülmez.
                        px = _safe_float(p_pos.get("map_x", p_pos.get("x", 0.0)))
                        py = _safe_float(p_pos.get("map_y", p_pos.get("y", 0.0)))
                    else:
                        px = _safe_float(p_pos[0]) if isinstance(p_pos, (tuple, list)) and len(p_pos) > 0 else 0.0
                        py = _safe_float(p_pos[1]) if isinstance(p_pos, (tuple, list)) and len(p_pos) > 1 else 0.0
                    heading = _radar_get("heading_deg", _radar_get("player_heading", _radar_get("heading", 0.0)))
                    t_bearing = _radar_get(
                        "target_bearing_deg",
                        _radar_get(
                            "target_bearing",
                            _radar_get("target_direction_deg", _radar_get("target_direction", None)),
                        ),
                    )
                    t_dist = _radar_get("target_distance_m", _radar_get("target_distance", _radar_get("target_dist", None)))
                    t_name = _radar_get("target_name", _radar_get("target_node", _radar_get("target", "")))
                    blips = _radar_get("radar_blips", _radar_get("blips", None))
                    r_range = _radar_get("radar_range_m", _radar_get("radar_range", None))

                    self._radar_projection.update_projection(
                        player_x=px,
                        player_y=py,
                        heading_deg=_safe_float(heading),
                        target_bearing_deg=(
                            _safe_float(t_bearing) if t_bearing is not None else None
                        ),
                        target_distance_m=(
                            _safe_float(t_dist) if t_dist is not None else None
                        ),
                        target_name=t_name,
                        blips=blips,
                        range_m=(_safe_float(r_range) if r_range is not None else None),
                    )

            # 8. Bağlı canlı kaynaklardan türetilen satırlar + düğme durumları.
            self._refresh_live_derived()
            self._sync_control_buttons()
            self._sync_mode_buttons()

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

    def _hud_is_ready(self) -> bool:
        """Gerçek hazırlık ölçümü: arayüz kuruldu mu, kart + rozet + başlık var mı?"""
        try:
            if self.__dict__.get("_ui_ready") is False:
                return False
            return bool(
                hasattr(self, "card")
                and hasattr(self, "badge_label")
                and hasattr(self, "title_label")
                and hasattr(self, "harvest_badge_label")
            )
        except Exception:
            return False

    def get_overlay_hud_health_summary(self) -> dict:
        try:
            ready = self._hud_is_ready()
            return {
                "overlay_hud_ready": ready,
                "ui_ready": bool(self.__dict__.get("_ui_ready", False)),
                "ui_init_error": self.__dict__.get("_ui_init_error"),
                "minimized": bool(self.__dict__.get("_minimized", False)),
            }
        except Exception:
            return {"overlay_hud_ready": False}

    def get_overlay_hud_ready_flag(self) -> dict:
        return {"module": "overlay_hud", "ready": self._hud_is_ready()}

    def get_overlay_hud_version_info(self) -> dict:
        return {"module": "overlay_hud", "version": HUD_VERSION, "ready": self._hud_is_ready()}

    def get_overlay_hud_diagnostic_snapshot(self) -> dict:
        ready = self._hud_is_ready()
        return {
            "module": "overlay_hud",
            "status": "ok" if ready else "degraded",
            "ready": ready,
            "ui_init_error": self.__dict__.get("_ui_init_error"),
        }


# NOT: GameOverlay takma adı burada YOKTUR — ui/overlay.py'deki GameOverlay
# tek GameOverlay sınıfıdır (iki farklı GameOverlay bulunması F-0676).
# Geriye dönük içe aktarma ui/overlay_window.py üzerinden sağlanır.

__all__ = [
    "OverlayHUD",
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
    """Sözleşmeyi gerçekten denetler: yetenek listesi boş olmamalı ve
    OverlayHUD, HUD'un vaat ettiği yöntemlerin hepsini taşımalı
    (update_state, harvest/minimap güncellemeleri, Seç-Topla, telemetri,
    tıklama geçişi, günlük). Hiçbir şeyi ölçmeden True dönmez."""
    try:
        if not get_overlay_hud_capabilities():
            return False
        required = (
            "update_state",
            "update_harvest_stage",
            "update_minimap_mapped",
            "update_consumable_and_selective_status",
            "set_selective_gather_status",
            "get_overlay_telemetry_snapshot",
            "set_click_through",
            "add_log",
            "set_mode",
            # Prototip uyumu (W5): rota/hedef/sistem/ayarlar sözleşmesi.
            "update_route_progress",
            "update_target",
            "update_system_status",
            "set_detection_boxes_visible",
            "set_minimap_visible",
            "reset_position",
            "bind_system_monitor",
            "bind_live_sources",
        )
        return all(callable(getattr(OverlayHUD, name, None)) for name in required)
    except Exception:
        return False


def _overlay_hud_contract_ok(_contract_ok: Optional[bool] = None) -> bool:
    """P2-1: zincir ici tek dogrulama (bkz. core/telemetry.py'deki aciklama).
    ``None`` ise dogrulama burada bir kez calisir; cagrilar arasi
    onbellek yoktur."""
    if _contract_ok is not None:
        return bool(_contract_ok)
    return bool(validate_overlay_hud_contract())


def get_overlay_hud_baseline_metrics(_contract_ok: Optional[bool] = None) -> dict:
    return {"module": "overlay_hud", "contract_valid": _overlay_hud_contract_ok(_contract_ok), "capability_count": len(get_overlay_hud_capabilities())}


def get_overlay_hud_readiness_status(_contract_ok: Optional[bool] = None) -> dict:
    try:
        ok = _overlay_hud_contract_ok(_contract_ok)
        return {"module": "overlay_hud", "ready": ok, "metrics": get_overlay_hud_baseline_metrics(_contract_ok=ok)}
    except Exception:
        return {"module": "overlay_hud", "ready": False, "metrics": {}}


def get_overlay_hud_audit_record(_contract_ok: Optional[bool] = None) -> dict:
    try:
        ok = _overlay_hud_contract_ok(_contract_ok)
        return {"module": "overlay_hud", "audit_passed": ok, "readiness": get_overlay_hud_readiness_status(_contract_ok=ok)}
    except Exception:
        return {"module": "overlay_hud", "audit_passed": False, "readiness": {}}


def get_overlay_hud_operational_digest(_contract_ok: Optional[bool] = None) -> dict:
    ok = _overlay_hud_contract_ok(_contract_ok)
    return {"module": "overlay_hud", "operational": ok, "audit": get_overlay_hud_audit_record(_contract_ok=ok)}


def get_overlay_hud_telemetry_payload(_contract_ok: Optional[bool] = None) -> dict:
    ok = _overlay_hud_contract_ok(_contract_ok)
    return {"module": "overlay_hud", "telemetry_ready": ok, "digest": get_overlay_hud_operational_digest(_contract_ok=ok)}


def get_overlay_hud_integrity_seal(_contract_ok: Optional[bool] = None) -> dict:
    ok = _overlay_hud_contract_ok(_contract_ok)
    return {"module": "overlay_hud", "sealed": ok, "telemetry": get_overlay_hud_telemetry_payload(_contract_ok=ok)}


def get_overlay_hud_compliance_certificate(_contract_ok: Optional[bool] = None) -> dict:
    ok = _overlay_hud_contract_ok(_contract_ok)
    return {"module": "overlay_hud", "certified": ok, "seal": get_overlay_hud_integrity_seal(_contract_ok=ok)}


def get_overlay_hud_readiness_probe(_contract_ok: Optional[bool] = None) -> dict:
    ok = _overlay_hud_contract_ok(_contract_ok)
    return {"module": "overlay_hud", "ready": ok, "certificate": get_overlay_hud_compliance_certificate(_contract_ok=ok)}


def get_overlay_hud_audit_summary(_contract_ok: Optional[bool] = None) -> dict:
    ok = _overlay_hud_contract_ok(_contract_ok)
    return {"module": "overlay_hud", "audit_passed": ok, "probe": get_overlay_hud_readiness_probe(_contract_ok=ok)}

