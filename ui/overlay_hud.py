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
- Görüntü işleme (Vision) çakışma kontrolleri için ekran dikdörtgeni geometrisi raporlama
"""

import ctypes
import logging
import re
import sys
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

from PyQt6.QtCore import QObject, QPoint, QRect, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QCursor, QFont, QMouseEvent, QPaintEvent, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
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
                    color: #e2e8f0;
                    border: 1px solid rgba(255, 255, 255, 0.20);
                    border-radius: 3px;
                    font-family: 'Segoe UI', 'Inter', sans-serif;
                    font-size: 9px;
                    font-weight: 700;
                    padding: 1px 4px;
                }
                """
            )

            self.desc_label = QLabel(self._description_text, self)
            self.desc_label.setStyleSheet(
                """
                QLabel {
                    background: transparent;
                    color: #94a3b8;
                    border: none;
                    font-family: 'Segoe UI', 'Inter', sans-serif;
                    font-size: 10px;
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


class OverlayHUD(QWidget):
    """
    Oyun İçi Şeffaf Durum Paneli (Overlay HUD — Türkçe).
    Piksel hassasiyetinde görsel tasarım, düğüm başına 3x toplama (1/3 -> 2/3 -> 3/3) aşama göstergesi,
    sağ üst Minimap otomatik haritalanan nokta sayacı ve tam durum/günlük telemetrisi sağlar.
    """

    f9_clicked = pyqtSignal()
    f8_clicked = pyqtSignal()
    harvest_stage_updated = pyqtSignal(int, int)
    minimap_mapped_updated = pyqtSignal(int)

    # Durum teması eşlemesi: (rozet_metni, vurgu_hex, rozet_arkaplan_hex, rozet_yazi_hex)
    STATE_THEMES: Dict[str, Tuple[str, str, str, str]] = {
        "WALKING": ("YÜRÜYOR", "#3b82f6", "#1d4ed8", "#eff6ff"),
        "YÜRÜYOR": ("YÜRÜYOR", "#3b82f6", "#1d4ed8", "#eff6ff"),
        "GATHERING": ("TOPLANIYOR", "#22c55e", "#15803d", "#f0fdf4"),
        "TOPLANIYOR": ("TOPLANIYOR", "#22c55e", "#15803d", "#f0fdf4"),
        "FIGHTING": ("SAVAŞTA", "#ef4444", "#b91c1c", "#ffffff"),
        "COMBAT": ("SAVAŞTA", "#ef4444", "#b91c1c", "#ffffff"),
        "SAVAŞTA": ("SAVAŞTA", "#ef4444", "#b91c1c", "#ffffff"),
        "PAUSED": ("DURAKLATILDI", "#64748b", "#334155", "#f1f5f9"),
        "DURAKLATILDI": ("DURAKLATILDI", "#64748b", "#334155", "#f1f5f9"),
        "WAITING": ("BEKLEMEDE", "#64748b", "#334155", "#f1f5f9"),
        "IDLE": ("BEKLEMEDE", "#64748b", "#334155", "#f1f5f9"),
        "BEKLEMEDE": ("BEKLEMEDE", "#64748b", "#334155", "#f1f5f9"),
        "BREAK": ("MOLA", "#f59e0b", "#b45309", "#ffffff"),
        "MOLA": ("MOLA", "#f59e0b", "#b45309", "#ffffff"),
        "MAP": ("HARİTA", "#f59e0b", "#b45309", "#ffffff"),
        "NAVIGATING": ("HARİTA", "#f59e0b", "#b45309", "#ffffff"),
        "HARİTA": ("HARİTA", "#f59e0b", "#b45309", "#ffffff"),
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
        self._click_through_enabled: bool = False
        self._stealth_ex_style: int = WS_EX_LAYERED | WS_EX_TOOLWINDOW

        self._init_window_properties()
        self._init_ui()
        self._init_timer()
        self._apply_state_theme("WAITING")
        self._refresh_harvest_badge_style()

    def _init_window_properties(self) -> None:
        """Çerçevesiz, her zaman üstte, araç penceresi bayraklarını ve yarı şeffaf özellikleri yapılandır."""
        try:
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            self.setFixedSize(430, 216)
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
            return True
        except Exception as exc:
            logger.error("OverlayHUD click-through durumu değiştirilemedi: %s", exc)
            return False

    def _init_ui(self) -> None:
        """3x Toplama Aşama Rozeti ve Sağ Üst Minimap Sayacı ile Türkçe HUD arayüzünü oluştur."""
        try:
            root_layout = QVBoxLayout(self)
            root_layout.setContentsMargins(0, 0, 0, 0)
            root_layout.setSpacing(0)

            # 3px sol vurgu kenarlıklı ve yarı şeffaf koyu arka planlı ana kart çerçevesi
            self.card = QFrame(self)
            self.card.setObjectName("OverlayCard")
            root_layout.addWidget(self.card)

            card_layout = QVBoxLayout(self.card)
            card_layout.setContentsMargins(14, 9, 14, 8)
            card_layout.setSpacing(3)

            # -----------------------------------------------------------------
            # 1. ÜST SATIR: Durum Rozeti | Mod Rozeti | 3x Toplama Rozeti | Esnek Boşluk | Süre
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
                    background-color: #272b33;
                    color: #cbd5e1;
                    border: 1px solid rgba(255, 255, 255, 0.14);
                    border-radius: 3px;
                    font-family: 'Segoe UI', 'Inter', sans-serif;
                    font-size: 9px;
                    font-weight: 700;
                    letter-spacing: 0.4px;
                    padding: 2px 6px;
                }
                """
            )

            # Düğüm Başına 3x Toplama aşama rozeti (ör. "3x Toplama (Harvest): 2/3")
            self.harvest_badge_label = QLabel("3x Toplama (Harvest: 0/3)", self.card)
            self.harvest_badge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.harvest_badge_lbl = self.harvest_badge_label

            self.elapsed_label = QLabel("0:00", self.card)
            self.elapsed_label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.elapsed_label.setStyleSheet(
                """
                QLabel {
                    background: transparent;
                    color: #94a3b8;
                    border: none;
                    font-family: 'Consolas', 'Segoe UI', monospace;
                    font-size: 11px;
                    font-weight: 500;
                }
                """
            )
            self.time_lbl = self.elapsed_label  # Geriye dönük uyumluluk takma adı

            header_layout.addWidget(self.badge_label)
            header_layout.addWidget(self.mode_label)
            header_layout.addWidget(self.harvest_badge_label)
            header_layout.addStretch(1)
            header_layout.addWidget(self.elapsed_label)
            card_layout.addLayout(header_layout)

            # -----------------------------------------------------------------
            # 2. ANA BAŞLIK, ALT BAŞLIK, 3X TOPLAMA İLERLEME ÇUBUĞU & AYIRICI ÇİZGİ
            # -----------------------------------------------------------------
            self.title_label = QLabel("Durum: Beklemede", self.card)
            self.title_label.setStyleSheet(
                """
                QLabel {
                    background: transparent;
                    color: #ffffff;
                    border: none;
                    font-family: 'Segoe UI', 'Inter', sans-serif;
                    font-size: 14px;
                    font-weight: 700;
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
                    color: #94a3b8;
                    border: none;
                    font-family: 'Segoe UI', 'Inter', sans-serif;
                    font-size: 11px;
                    font-weight: 400;
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
                    background-color: rgba(255, 255, 255, 0.08);
                    border: none;
                    border-radius: 2px;
                }
                QProgressBar::chunk {
                    background-color: #22c55e;
                    border-radius: 2px;
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
            stats_grid.setContentsMargins(0, 2, 0, 2)
            stats_grid.setHorizontalSpacing(16)
            stats_grid.setVerticalSpacing(1)

            self.stat1_title = QLabel("TOPLANAN", self.stats_container)
            self.stat1_value = QLabel("Toplanan: 0", self.stats_container)

            self.stat2_title = QLabel("AKTİF / DİNLENEN / ROTA", self.stats_container)
            self.stat2_value = QLabel("0 aktif · 0 dinlenen", self.stats_container)

            self.stat3_title = QLabel("ARD ARDA BAŞARISIZ", self.stats_container)
            self.stat3_value = QLabel("0 / 10", self.stats_container)

            # Geriye dönük uyumluluk takma adları
            self.stat_g_title = self.stat1_title
            self.stat_g_val = self.stat1_value
            self.stat_f_title = self.stat2_title
            self.stat_f_val = self.stat2_value
            self.stat_m_title = self.stat3_title
            self.stat_m_val = self.stat3_value

            header_col_style = """
                QLabel {
                    background: transparent;
                    color: #64748b;
                    border: none;
                    font-family: 'Segoe UI', 'Inter', sans-serif;
                    font-size: 9px;
                    font-weight: 700;
                    letter-spacing: 0.3px;
                }
            """
            value_col_style = """
                QLabel {
                    background: transparent;
                    color: #ffffff;
                    border: none;
                    font-family: 'Segoe UI', 'Inter', sans-serif;
                    font-size: 13px;
                    font-weight: 700;
                }
            """

            for header_lbl in (self.stat1_title, self.stat2_title, self.stat3_title):
                header_lbl.setStyleSheet(header_col_style)
                header_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            for val_lbl in (self.stat1_value, self.stat2_value, self.stat3_value):
                val_lbl.setStyleSheet(value_col_style)
                val_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            stats_grid.addWidget(self.stat1_title, 0, 0)
            stats_grid.addWidget(self.stat1_value, 1, 0)
            stats_grid.addWidget(self.stat2_title, 0, 1)
            stats_grid.addWidget(self.stat2_value, 1, 1)
            stats_grid.addWidget(self.stat3_title, 0, 2)
            stats_grid.addWidget(self.stat3_value, 1, 2)
            stats_grid.setColumnStretch(0, 2)
            stats_grid.setColumnStretch(1, 5)
            stats_grid.setColumnStretch(2, 3)

            card_layout.addWidget(self.stats_container)

            # -----------------------------------------------------------------
            # 4. SON 3 ZAMAN DAMGALI GÜNLÜK (LOG) SATIRI
            # -----------------------------------------------------------------
            self.logs_container = QWidget(self.card)
            self.logs_container.setStyleSheet("QWidget { background: transparent; border: none; }")
            logs_layout = QVBoxLayout(self.logs_container)
            logs_layout.setContentsMargins(0, 1, 0, 1)
            logs_layout.setSpacing(1)

            log_line_style = """
                QLabel {
                    background: transparent;
                    color: #9ca3af;
                    border: none;
                    font-family: 'Consolas', 'Segoe UI', monospace;
                    font-size: 10px;
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
                log_lbl.setFixedHeight(13)
                logs_layout.addWidget(log_lbl)

            card_layout.addWidget(self.logs_container)
            card_layout.addStretch(1)

            # -----------------------------------------------------------------
            # 5. ALT BİLGİ SATIRI: [F9] Menü | [F8] Nokta Ekle | Haritalanan: X | Sürüm
            # -----------------------------------------------------------------
            footer_layout = QHBoxLayout()
            footer_layout.setContentsMargins(0, 2, 0, 0)
            footer_layout.setSpacing(10)

            self.f9_shortcut = ClickableShortcutWidget("F9", "Menü", self.card)
            self.f9_shortcut.setToolTip("F9: Hızlı Kontrol Menüsünü Aç")
            self.f9_shortcut.clicked.connect(self._on_f9_clicked)

            self.f8_shortcut = ClickableShortcutWidget("F8", "Nokta Ekle", self.card)
            self.f8_shortcut.setToolTip("F8: Mevcut Konumu Rota Noktası Olarak Kaydet")
            self.f8_shortcut.clicked.connect(self._on_f8_clicked)

            # Sağ üst Minimap'ten otomatik kaydedilen canlı materyal noktası sayacı
            self.minimap_mapped_label = QLabel("Haritalanan: 0 (Minimap Mapped: 0)", self.card)
            self.minimap_mapped_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.minimap_mapped_label.setToolTip("Sağ üst Minimap'ten otomatik haritalanan kaynak noktası sayısı")
            self.minimap_mapped_label.setStyleSheet(
                """
                QLabel {
                    background-color: rgba(56, 189, 248, 0.12);
                    color: #38bdf8;
                    border: 1px solid rgba(56, 189, 248, 0.30);
                    border-radius: 3px;
                    font-family: 'Segoe UI', 'Inter', sans-serif;
                    font-size: 9px;
                    font-weight: 700;
                    padding: 1px 6px;
                }
                """
            )
            self.minimap_mapped_lbl = self.minimap_mapped_label

            self.version_label = QLabel("v1.6.15", self.card)
            self.version_label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.version_label.setStyleSheet(
                """
                QLabel {
                    background: transparent;
                    color: #64748b;
                    border: none;
                    font-family: 'Segoe UI', 'Inter', sans-serif;
                    font-size: 10px;
                    font-weight: 500;
                }
                """
            )

            footer_layout.addWidget(self.f9_shortcut)
            footer_layout.addWidget(self.f8_shortcut)
            footer_layout.addWidget(self.minimap_mapped_label)
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
            if is_active:
                self.harvest_badge_label.setStyleSheet(
                    """
                    QLabel {
                        background-color: rgba(34, 197, 94, 0.22);
                        color: #4ade80;
                        border: 1px solid rgba(34, 197, 94, 0.50);
                        border-radius: 3px;
                        font-family: 'Segoe UI', 'Inter', sans-serif;
                        font-size: 9px;
                        font-weight: 700;
                        letter-spacing: 0.3px;
                        padding: 2px 6px;
                    }
                    """
                )
            else:
                self.harvest_badge_label.setStyleSheet(
                    """
                    QLabel {
                        background-color: rgba(255, 255, 255, 0.06);
                        color: #94a3b8;
                        border: 1px solid rgba(255, 255, 255, 0.14);
                        border-radius: 3px;
                        font-family: 'Segoe UI', 'Inter', sans-serif;
                        font-size: 9px;
                        font-weight: 700;
                        letter-spacing: 0.3px;
                        padding: 2px 6px;
                    }
                    """
                )
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
            self.minimap_mapped_label.setText(
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
                    border: none;
                    border-radius: 3px;
                    font-family: 'Segoe UI', 'Inter', sans-serif;
                    font-size: 9px;
                    font-weight: 700;
                    letter-spacing: 0.4px;
                    padding: 2px 6px;
                }}
                """
            )

            self.card.setStyleSheet(
                f"""
                QFrame#OverlayCard {{
                    background-color: rgba(20, 23, 28, 220);
                    border-top: 1px solid rgba(255, 255, 255, 0.08);
                    border-right: 1px solid rgba(255, 255, 255, 0.08);
                    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
                    border-left: 3px solid {accent_hex};
                    border-radius: 4px;
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

    def get_overlay_hud_health_summary(self) -> dict:
        try:
            return {"overlay_hud_ready": True}
        except Exception:
            return {"overlay_hud_ready": False}

    def get_overlay_hud_ready_flag(self) -> dict:
        return {"module": "overlay_hud", "ready": True}

    def get_overlay_hud_version_info(self) -> dict:
        return {"module": "overlay_hud", "version": "2.0.0", "ready": True}


# GameOverlay içe aktaran modüller için geriye dönük uyumluluk takma adı
GameOverlay = OverlayHUD

__all__ = ["OverlayHUD", "GameOverlay", "OverlaySignals", "ClickableShortcutWidget"]
