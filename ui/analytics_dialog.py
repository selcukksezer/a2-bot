# -*- coding: utf-8 -*-
import csv
import json
import logging
import os
import re
import time
import weakref
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.telemetry import TelemetryEngine
from core.economy_broker import SessionProfitSimulator, format_kinah_tr
from core.lifetime_stats import get_lifetime_stats
from ui.shell_page_frame import wrap_shell_tab_root_layout
from version import __version__

logger = logging.getLogger("AnalyticsDialog")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [AnalyticsUI] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class _QtLogHandler(logging.Handler):
    """Bot logger'larından gelen gerçek kayıtları dialog'a taşır.

    Kayıtlar başka thread'lerden gelebilir; handler yalnızca dialog'un
    pyqtSignal'ını tetikler (queued bağlantı GUI thread'inde çalışır).
    Dialog silinmişse weakref boş döner ve kayıt sessizce atılır.
    """

    def __init__(self, dialog_ref: Any) -> None:
        super().__init__(level=logging.INFO)
        self._dialog_ref = dialog_ref

    def emit(self, record: logging.LogRecord) -> None:  # noqa: N802
        try:
            dlg = self._dialog_ref()
            if dlg is None:
                return
            dlg.log_record_received.emit(record.levelname, record.getMessage())
        except Exception:
            pass


# Canlı Günlük'e bağlanan logger adları: botun gerçek kayıt kaynakları.
_LIVE_LOG_SOURCES = ("Gatherion", "AnalyticsDialog", "TelemetryEngine", "EconomyBroker")

# logs/bot.log satır biçimi: [21:15:58] [Gatherion.StateMachine] [INFO] mesaj
_LOG_LINE_RE = re.compile(
    r"^\[(?P<time>\d{2}:\d{2}:\d{2})\] \[(?P<name>[^\]]+)\] "
    r"\[(?P<level>INFO|WARNING|ERROR|DEBUG|CRITICAL)\] (?P<msg>.*)$"
)


class KPICard(QFrame):
    """Üst özet göstergeleri için istatistik.html `.scard` tasarımına birebir uyan KPI Kartı."""

    def __init__(
        self,
        title: str,
        initial_value: str = "0",
        subtext: str = "",
        accent_color: str = "#56c8f5",
        compact: bool = False,
        with_progress: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.accent_color = accent_color
        self.setObjectName("KPICard")
        self.setMinimumHeight(80 if compact else 96)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            f"""
            QFrame#KPICard {{
                background-color: #11151c;
                border: 1px solid #1e2530;
                border-top: 3px solid {self.accent_color};
                border-radius: 9px;
            }}
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(13, 10, 13, 10)
        layout.setSpacing(3)

        self.title_lbl = QLabel(title.upper(), self)
        self.title_lbl.setStyleSheet(
            "color: #9aa3af; font-size: 11px; font-weight: 800; letter-spacing: 0.7px; border: none; background: transparent;"
        )

        val_size = 20 if compact else 24
        self.value_lbl = QLabel(initial_value, self)
        self.value_lbl.setStyleSheet(
            f'color: #eef1f5; font-size: {val_size}px; font-weight: 800; font-family: Consolas, "Cascadia Mono", monospace; border: none; background: transparent;'
        )

        self.sub_lbl = QLabel(subtext, self)
        self.sub_lbl.setStyleSheet(
            "color: #9aa3af; font-size: 12px; font-weight: 500; border: none; background: transparent;"
        )

        layout.addWidget(self.title_lbl)
        layout.addWidget(self.value_lbl)
        layout.addWidget(self.sub_lbl)

        self.progress_bar: Optional[QProgressBar] = None
        if with_progress:
            pb = QProgressBar(self)
            pb.setRange(0, 100)
            pb.setValue(0)
            pb.setTextVisible(False)
            pb.setFixedHeight(5)
            pb.setStyleSheet(
                """
                QProgressBar {
                    background-color: #0d1015;
                    border: 1px solid #1e2530;
                    border-radius: 2px;
                }
                QProgressBar::chunk {
                    background-color: #34d399;
                    border-radius: 2px;
                }
                """
            )
            self.progress_bar = pb
            layout.addWidget(pb)

    def update_card(self, value_str: str, subtext: str = "", progress_pct: Optional[int] = None) -> None:
        self.value_lbl.setText(str(value_str))
        if subtext:
            self.sub_lbl.setText(str(subtext))
        if self.progress_bar is not None and progress_pct is not None:
            self.progress_bar.setValue(max(0, min(100, int(progress_pct))))


class _InsightCard(QFrame):
    """Üçlü içgörü şeridi (`.insights > .icard`) bileşeni."""

    def __init__(
        self,
        tag_text: str,
        initial_val: str = "—",
        initial_sub: str = "—",
        tag_color: str = "#6b7280",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("InsightCard")
        self.setStyleSheet(
            """
            QFrame#InsightCard {
                background-color: #11151c;
                border: 1px solid #1e2530;
                border-radius: 8px;
            }
            """
        )
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(10)

        self.tag_lbl = QLabel(tag_text.upper(), self)
        self.tag_lbl.setStyleSheet(
            f"color: {tag_color}; font-size: 10px; font-weight: 800; letter-spacing: 0.9px; border: none; background: transparent;"
        )
        row.addWidget(self.tag_lbl)

        self.val_lbl = QLabel(initial_val, self)
        self.val_lbl.setWordWrap(True)
        self.val_lbl.setStyleSheet(
            "color: #eef1f5; font-size: 13px; font-weight: 700; border: none; background: transparent;"
        )
        row.addWidget(self.val_lbl)

        row.addStretch(1)

        self.sub_lbl = QLabel(initial_sub, self)
        self.sub_lbl.setStyleSheet(
            'color: #9aa3af; font-family: Consolas, "Cascadia Mono", monospace; font-size: 12px; border: none; background: transparent;'
        )
        row.addWidget(self.sub_lbl)

    def update_insight(self, val_str: str, sub_str: str) -> None:
        self.val_lbl.setText(str(val_str))
        self.sub_lbl.setText(str(sub_str))


class _ShareBarCellWidget(QWidget):
    """Tablodaki `PAY %` sütunu için `.pbar` (56x5px gök mavisi çubuk + yüzde metni)."""

    def __init__(self, pct: float, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pct = max(0.0, min(100.0, float(pct)))
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(6)

        bar = QProgressBar(self)
        bar.setRange(0, 100)
        bar.setValue(int(round(self._pct)))
        bar.setTextVisible(False)
        bar.setFixedSize(64, 6)
        bar.setStyleSheet(
            """
            QProgressBar {
                background-color: #0d1015;
                border: 1px solid #1e2530;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background-color: #56c8f5;
                border-radius: 2px;
            }
            """
        )
        row.addWidget(bar)

        lbl = QLabel(f"%{self._pct:.1f}", self)
        lbl.setStyleSheet(
            'color: #eef1f5; font-family: Consolas, "Cascadia Mono", monospace; font-size: 12px; background: transparent; border: none;'
        )
        row.addWidget(lbl)
        row.addStretch(1)


class LiveThroughputChart(QWidget):
    """
    QPainter ile çizilen özel Canlı Verim Grafiği (`istatistik.html` `#chart` ile birebir).
    12 zaman dilimine göre toplanan materyal sayısını gök mavisi (`#56c8f5`) çubuk,
    saatlik hızı zümrüt yeşili (`#34d399`) çizgi + halka nokta olarak çizer.
    Dilime tıklanarak filtreleme yapılmasını destekler.
    """

    slice_clicked = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(210)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.buckets: List[Dict[str, Any]] = []
        self.selected_slice: Optional[int] = None
        self.active_time_filter: str = "Tümü"
        self._plot_rect = QRectF()

    @staticmethod
    def _safe_int(val: Any, default: int = 0) -> int:
        try:
            if val is None:
                return default
            return int(float(val))
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        try:
            if val is None:
                return default
            res = float(val)
            if res != res or res in (float("inf"), float("-inf")):
                return default
            return res
        except (ValueError, TypeError):
            return default

    def set_time_filter(self, time_filter: str) -> None:
        """Zaman filtresi ('1s', '6s', '24s', 'Tümü' vb.) geçişlerinde dilim seçimini güvenli sıfırlar."""
        norm = str(time_filter or "Tümü").strip() or "Tümü"
        if self.active_time_filter != norm:
            self.active_time_filter = norm
            self.selected_slice = None
            self.update()

    def set_buckets(self, buckets: Optional[List[Dict[str, Any]]]) -> None:
        raw = list(buckets) if isinstance(buckets, (list, tuple)) else []
        cleaned: List[Dict[str, Any]] = []
        for idx, b in enumerate(raw):
            if isinstance(b, dict):
                cnt = max(0, self._safe_int(b.get("count", 0), 0))
                rate = max(0.0, self._safe_float(b.get("rate_per_hour", 0.0), 0.0))
                lbl = str(b.get("label", f"#{idx + 1}") or f"#{idx + 1}")
                item = dict(b)
                item["count"] = cnt
                item["rate_per_hour"] = rate
                item["label"] = lbl
                cleaned.append(item)
        self.buckets = cleaned
        if self.selected_slice is not None:
            if not self.buckets or self.selected_slice < 0 or self.selected_slice >= len(self.buckets):
                self.selected_slice = None
        self.update()

    def set_selected_slice(self, idx: Optional[int]) -> None:
        if idx is None or not self.buckets or idx < 0 or idx >= len(self.buckets):
            normalized: Optional[int] = None
        else:
            normalized = int(idx)
        if self.selected_slice != normalized:
            self.selected_slice = normalized
            self.update()

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        try:
            if (
                event is not None
                and event.button() == Qt.MouseButton.LeftButton
                and self.buckets
                and not self._plot_rect.isEmpty()
                and self._plot_rect.width() > 1.0
            ):
                pos = event.position()
                if self._plot_rect.contains(pos):
                    n = max(1, len(self.buckets))
                    slot_w = max(1e-6, self._plot_rect.width() / float(n))
                    idx = int((pos.x() - self._plot_rect.left()) / slot_w)
                    idx = max(0, min(n - 1, idx))
                    self.slice_clicked.emit(idx)
                    return
        except Exception as exc:
            logger.debug(f"LiveThroughputChart mousePressEvent koruma: {exc}")
        super().mousePressEvent(event)

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

            rect = self.rect()
            if rect.width() <= 2 or rect.height() <= 2:
                return
            bg_rect = QRectF(0.5, 0.5, max(1.0, rect.width() - 1.0), max(1.0, rect.height() - 1.0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#11151c")))
            painter.drawRoundedRect(bg_rect, 8.0, 8.0)

            margin_left = 46.0
            margin_right = 52.0
            margin_top = 12.0
            margin_bottom = 28.0

            plot_w = max(40.0, rect.width() - margin_left - margin_right)
            plot_h = max(40.0, rect.height() - margin_top - margin_bottom)
            plot_rect = QRectF(margin_left, margin_top, plot_w, plot_h)
            self._plot_rect = plot_rect

            grid_pen = QPen(QColor(255, 255, 255, 16), 1.0, Qt.PenStyle.DashLine)
            grid_pen.setDashPattern([3.0, 3.0])
            label_font = QFont("Consolas", 9)
            painter.setFont(label_font)

            valid_buckets = [b for b in (self.buckets or []) if isinstance(b, dict)]
            counts = [max(0, self._safe_int(b.get("count", 0), 0)) for b in valid_buckets] if valid_buckets else [0]
            rates = [max(0.0, self._safe_float(b.get("rate_per_hour", 0.0), 0.0)) for b in valid_buckets] if valid_buckets else [0.0]
            max_count = max(5, max(counts) if counts else 5)
            max_rate = max(25.0, max(rates) if rates else 25.0)
            denom_count = float(max(1, max_count))
            denom_rate = float(max(1e-6, max_rate))

            steps = 4
            denom_steps = float(max(1, steps))
            for i in range(steps + 1):
                ratio = i / denom_steps
                y = plot_rect.bottom() - ratio * plot_h
                painter.setPen(grid_pen)
                painter.drawLine(QPointF(plot_rect.left(), y), QPointF(plot_rect.right(), y))

                val_count = int(round(ratio * max_count))
                val_rate = int(round(ratio * max_rate))
                painter.setPen(QColor("#6b7280"))
                painter.drawText(
                    QRectF(2.0, y - 8.0, margin_left - 8.0, 16.0),
                    int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                    f"{val_count}",
                )
                painter.setPen(QColor("#34d399"))
                painter.drawText(
                    QRectF(plot_rect.right() + 6.0, y - 8.0, margin_right - 8.0, 16.0),
                    int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                    f"{val_rate}/sa",
                )

            if not valid_buckets:
                painter.setPen(QColor("#6b7280"))
                painter.drawText(plot_rect, int(Qt.AlignmentFlag.AlignCenter), "Bu aralıkta gösterilecek kayıt yok — henüz kayıt yok")
                return

            n = max(1, len(valid_buckets))
            slot_w = max(1.0, plot_w / float(n))
            bar_w = min(28.0, max(8.0, slot_w * 0.56))

            line_points: List[QPointF] = []

            for idx, b in enumerate(valid_buckets):
                cnt = max(0, self._safe_int(b.get("count", 0), 0))
                rate = max(0.0, self._safe_float(b.get("rate_per_hour", 0.0), 0.0))
                cx = plot_rect.left() + (idx + 0.5) * slot_w
                is_sel = self.selected_slice == idx

                if is_sel:
                    sel_rect = QRectF(
                        plot_rect.left() + idx * slot_w + 2.0,
                        plot_rect.top(),
                        max(2.0, slot_w - 4.0),
                        plot_h,
                    )
                    sel_pen = QPen(QColor(245, 181, 10, 140), 1.0, Qt.PenStyle.DashLine)
                    sel_pen.setDashPattern([2.0, 2.0])
                    painter.setPen(sel_pen)
                    painter.setBrush(QColor(245, 181, 10, 26))
                    painter.drawRoundedRect(sel_rect, 4.0, 4.0)

                bar_h = min(plot_h, max(0.0, (cnt / denom_count) * plot_h))
                if bar_h > 0.5:
                    bar_rect = QRectF(cx - bar_w / 2.0, plot_rect.bottom() - bar_h, bar_w, max(2.0, bar_h))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor("#f5b50a") if is_sel else QColor(86, 200, 245, 215))
                    painter.drawRoundedRect(bar_rect, 3.0, 3.0)

                ly = plot_rect.bottom() - min(plot_h, max(0.0, (rate / denom_rate) * plot_h))
                line_points.append(QPointF(cx, ly))

                lbl = str(b.get("label", ""))
                painter.setPen(QColor("#f5b50a") if is_sel else QColor("#9aa3af"))
                painter.drawText(
                    QRectF(cx - 26.0, plot_rect.bottom() + 5.0, 52.0, 18.0),
                    int(Qt.AlignmentFlag.AlignCenter),
                    lbl,
                )

            if len(line_points) >= 2:
                line_path = QPainterPath()
                line_path.moveTo(line_points[0])
                for pt in line_points[1:]:
                    line_path.lineTo(pt)

                painter.setPen(QPen(QColor("#34d399"), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(line_path)

                for idx, pt in enumerate(line_points):
                    is_sel = self.selected_slice == idx
                    painter.setPen(QPen(QColor("#f5b50a" if is_sel else "#34d399"), 2.0))
                    painter.setBrush(QBrush(QColor("#f5b50a" if is_sel else "#0d1015")))
                    r = 4.5 if is_sel else 3.2
                    painter.drawEllipse(pt, r, r)
        except Exception as exc:
            logger.error(f"LiveThroughputChart çizim hatası: {exc}", exc_info=True)
        finally:
            painter.end()


class AnalyticsDialog(QDialog):
    """
    `C:\\Users\\selçuk\\Desktop\\istatistik.html` prototipine birebir (pixel-perfect) uyan
    Canlı Performans & İstatistik Paneli.
    """

    # Canlı Günlük: bot logger'larından gelen gerçek kayıtlar (thread-safe).
    log_record_received = pyqtSignal(str, str)

    def __init__(
        self,
        telemetry: Optional[TelemetryEngine] = None,
        parent: Optional[QWidget] = None,
        telemetry_engine: Optional[TelemetryEngine] = None,
        economy_calculator: Optional[Any] = None,
        economy_broker: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent)
        resolved_tel = telemetry if telemetry is not None else telemetry_engine
        self.telemetry = resolved_tel if resolved_tel is not None else TelemetryEngine()

        resolved_econ = economy_calculator if economy_calculator is not None else economy_broker
        if resolved_econ is not None:
            if hasattr(resolved_econ, "get_session_economy_summary") and callable(resolved_econ.get_session_economy_summary):
                self.economy_calculator = resolved_econ
            elif hasattr(resolved_econ, "scan_broker_window") or hasattr(resolved_econ, "get_price"):
                self.economy_calculator = SessionProfitSimulator(price_scanner=resolved_econ)
            else:
                self.economy_calculator = SessionProfitSimulator()
        else:
            self.economy_calculator = SessionProfitSimulator()

        self._active_range = "today"
        self._selected_slice: Optional[int] = None
        self._selected_group_key: Optional[str] = None
        self._selected_event: Optional[Tuple[str, Dict[str, Any]]] = None
        self._sort_col = "count"
        self._sort_dir = -1
        self._last_rows: List[Dict[str, Any]] = []
        self._last_metrics: Dict[str, Any] = {}
        self._updating_filters = False
        # Canlı Günlük durumu: (saat, seviye, mesaj) kayıtları, en fazla 300.
        self._log_records: deque = deque(maxlen=300)
        self._log_filter = "all"
        self._log_handler: Optional[_QtLogHandler] = None

        self.setWindowTitle("Telemetri ve Performans Analizi")
        self.resize(1120, 820)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #0a0c10;
                color: #eef1f5;
                font-family: 'Segoe UI', system-ui, sans-serif;
            }
            QLabel {
                color: #eef1f5;
                background: transparent;
            }
            QScrollArea#StatsScroll, QWidget#StatsScrollContent {
                background-color: #0a0c10;
                border: none;
            }
            QFrame#StatsToolbar, QFrame#StatsChartCard, QFrame#StatsDetailCard, QFrame#StatsTableCard {
                background-color: #11151c;
                border: 1px solid #1e2530;
                border-radius: 9px;
            }
            QComboBox, QSpinBox {
                background-color: #0d1015;
                color: #eef1f5;
                border: 1px solid #1e2530;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 12px;
                font-weight: 600;
                min-height: 26px;
            }
            QComboBox:hover, QSpinBox:hover {
                border-color: #2c3545;
            }
            QComboBox::drop-down {
                border: none;
                width: 18px;
            }
            QComboBox QAbstractItemView {
                background-color: #0d1015;
                color: #eef1f5;
                border: 1px solid #1e2530;
                selection-background-color: #1f2733;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 0px;
                border: none;
            }
            QPushButton[rangeSeg="true"] {
                background: transparent;
                color: #9aa3af;
                border: none;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: 700;
                border-radius: 5px;
            }
            QPushButton[rangeSeg="true"]:hover {
                color: #eef1f5;
            }
            QPushButton[rangeSeg="true"]:checked {
                background-color: #f5b50a;
                color: #120d00;
            }
            QPushButton {
                background-color: #11151c;
                color: #eef1f5;
                border: 1px solid #1e2530;
                border-radius: 7px;
                padding: 7px 13px;
                font-size: 12px;
                font-weight: 700;
            }
            QPushButton:hover {
                background-color: #171d27;
                border-color: #2c3545;
            }
            QTableWidget {
                background-color: #11151c;
                alternate-background-color: #131820;
                color: #eef1f5;
                gridline-color: transparent;
                border: none;
                font-size: 13px;
                selection-background-color: rgba(245, 181, 10, 0.14);
                selection-color: #eef1f5;
            }
            QHeaderView::section {
                background-color: #0d1015;
                color: #9aa3af;
                padding: 8px 10px;
                border: none;
                border-bottom: 1px solid #1e2530;
                font-weight: 800;
                font-size: 11px;
                letter-spacing: 0.6px;
            }
            QHeaderView::section:hover {
                color: #eef1f5;
            }
            """
        )

        self._build_ui()
        self._setup_live_log()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(1000)
        self._refresh_timer.timeout.connect(self.refresh_metrics)
        self._refresh_timer.start()

        self.refresh_metrics()

    def _build_ui(self) -> None:
        root_layout = wrap_shell_tab_root_layout(self, "stats")
        root_layout.setSpacing(10)

        scroll = QScrollArea(self)
        scroll.setObjectName("StatsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget(scroll)
        content.setObjectName("StatsScrollContent")
        page_lay = QVBoxLayout(content)
        page_lay.setContentsMargins(6, 2, 6, 4)
        page_lay.setSpacing(10)

        # 1. HEADER (.head)
        head_row = QHBoxLayout()
        head_row.setContentsMargins(0, 0, 0, 0)
        head_row.setSpacing(12)

        left_titles = QVBoxLayout()
        left_titles.setContentsMargins(0, 0, 0, 0)
        left_titles.setSpacing(1)
        kicker = QLabel("ÖLÇÜM", content)
        kicker.setStyleSheet(
            "color: #9aa3af; font-size: 11px; font-weight: 800; letter-spacing: 2.0px;"
        )
        title = QLabel("İstatistik", content)
        title.setStyleSheet("color: #eef1f5; font-size: 22px; font-weight: 800;")
        sub = QLabel(
            "Kayıtlı toplama olayları üzerinden hesaplanır — filtreler tüm panelleri günceller.",
            content,
        )
        sub.setStyleSheet("color: #9aa3af; font-size: 13px;")
        left_titles.addWidget(kicker)
        left_titles.addWidget(title)
        left_titles.addWidget(sub)
        head_row.addLayout(left_titles)
        head_row.addStretch(1)

        right_head = QVBoxLayout()
        right_head.setContentsMargins(0, 0, 0, 0)
        right_head.setSpacing(4)
        right_head.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.session_info_lbl = QLabel("Oturum: 00:00:00 | Savaşlar: 0 | Hatalar: 0", content)
        self.session_info_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.session_info_lbl.setStyleSheet(
            'font-size: 12px; font-weight: 600; color: #f5b50a; font-family: Consolas, "Cascadia Mono", monospace;'
        )
        right_head.addWidget(self.session_info_lbl)

        self.lifetime_info_lbl = QLabel("Kalıcı: Materyal 0 · Altın 0 · Canavar 0", content)
        self.lifetime_info_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.lifetime_info_lbl.setStyleSheet(
            'font-size: 12px; font-weight: 600; color: #56c8f5; font-family: Consolas, "Cascadia Mono", monospace;'
        )
        right_head.addWidget(self.lifetime_info_lbl)

        acts_row = QHBoxLayout()
        acts_row.setContentsMargins(0, 0, 0, 0)
        acts_row.setSpacing(6)
        acts_row.addStretch(1)

        hiz_lbl = QLabel("HIZ", content)
        hiz_lbl.setStyleSheet("color: #9aa3af; font-size: 11px; font-weight: 700; letter-spacing: 0.6px;")
        acts_row.addWidget(hiz_lbl)

        self.speed_combo = QComboBox(content)
        self.speed_combo.addItem("Yavaş", 2600)
        self.speed_combo.addItem("Normal", 1500)
        self.speed_combo.addItem("Hızlı", 700)
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        acts_row.addWidget(self.speed_combo)

        self.btn_live = QPushButton("Canlı Akışı Duraklat", content)
        self.btn_live.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_live.clicked.connect(self._toggle_live_timer)
        acts_row.addWidget(self.btn_live)

        self.btn_refresh = QPushButton("Yenile", content)
        self.btn_refresh.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_refresh.clicked.connect(self.refresh_metrics)
        acts_row.addWidget(self.btn_refresh)
        right_head.addLayout(acts_row)

        head_row.addLayout(right_head)
        page_lay.addLayout(head_row)

        # 2. FILTER TOOLBAR (.toolbar)
        toolbar = QFrame(content)
        toolbar.setObjectName("StatsToolbar")
        tb_lay = QHBoxLayout(toolbar)
        tb_lay.setContentsMargins(14, 9, 14, 9)
        tb_lay.setSpacing(10)

        seg_frame = QFrame(toolbar)
        seg_frame.setStyleSheet(
            "QFrame { background-color: #0d1015; border: 1px solid #1e2530; border-radius: 7px; }"
        )
        seg_lay = QHBoxLayout(seg_frame)
        seg_lay.setContentsMargins(2, 2, 2, 2)
        seg_lay.setSpacing(2)
        self._range_group = QButtonGroup(toolbar)
        self._range_group.setExclusive(True)
        self._range_buttons: Dict[str, QPushButton] = {}
        for r_key, r_label in (
            ("session", "Oturum"),
            ("today", "Bugün"),
            ("7d", "Son 7 Gün"),
            ("all", "Kalıcı"),
        ):
            b = QPushButton(r_label, seg_frame)
            b.setProperty("rangeSeg", True)
            b.setCheckable(True)
            b.setChecked(r_key == "today")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _c=False, rk=r_key: self._on_range_selected(rk))
            self._range_group.addButton(b)
            self._range_buttons[r_key] = b
            seg_lay.addWidget(b)
        tb_lay.addWidget(seg_frame)

        lbl_b = QLabel("BÖLGE", toolbar)
        lbl_b.setStyleSheet("color: #9aa3af; font-size: 11px; font-weight: 700; letter-spacing: 0.6px;")
        tb_lay.addWidget(lbl_b)
        self.region_filter = QComboBox(toolbar)
        self.region_filter.addItem("Tümü", "ALL")
        self.region_filter.currentIndexChanged.connect(self._on_filter_changed)
        tb_lay.addWidget(self.region_filter)

        lbl_r = QLabel("ROTA", toolbar)
        lbl_r.setStyleSheet("color: #9aa3af; font-size: 11px; font-weight: 700; letter-spacing: 0.6px;")
        tb_lay.addWidget(lbl_r)
        self.route_filter = QComboBox(toolbar)
        self.route_filter.addItem("Tümü", "ALL")
        self.route_filter.currentIndexChanged.connect(self._on_filter_changed)
        tb_lay.addWidget(self.route_filter)

        lbl_t = QLabel("HEDEF /SA", toolbar)
        lbl_t.setStyleSheet("color: #9aa3af; font-size: 11px; font-weight: 700; letter-spacing: 0.6px;")
        tb_lay.addWidget(lbl_t)
        self.target_rate_spin = QSpinBox(toolbar)
        self.target_rate_spin.setRange(5, 500)
        self.target_rate_spin.setValue(40)
        self.target_rate_spin.setFixedWidth(64)
        self.target_rate_spin.valueChanged.connect(lambda _v: self.refresh_metrics())
        tb_lay.addWidget(self.target_rate_spin)

        self.slice_chip_btn = QPushButton("Dilim: — ×", toolbar)
        self.slice_chip_btn.setVisible(False)
        self.slice_chip_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.slice_chip_btn.setStyleSheet(
            "background: rgba(245, 181, 10, 0.14); border: 1px solid rgba(245, 181, 10, 0.45); color: #f5b50a; padding: 4px 10px; border-radius: 99px; font-size: 12px; font-weight: 700;"
        )
        self.slice_chip_btn.clicked.connect(lambda: self._on_chart_slice_clicked(-1))
        tb_lay.addWidget(self.slice_chip_btn)

        tb_lay.addStretch(1)

        self.rec_count_lbl = QLabel('<b style="color:#eef1f5;">0</b> kayıt · BUGÜN', toolbar)
        self.rec_count_lbl.setTextFormat(Qt.TextFormat.RichText)
        self.rec_count_lbl.setStyleSheet(
            'color: #6b7280; font-family: Consolas, "Cascadia Mono", monospace; font-size: 12px;'
        )
        tb_lay.addWidget(self.rec_count_lbl)
        page_lay.addWidget(toolbar)

        # 3. KALICI İSTATİSTİKLER (ömür boyu)
        lifetime_title = QLabel("KALICI İSTATİSTİKLER (ÖMÜR BOYU)", content)
        lifetime_title.setStyleSheet(
            "font-size: 11px; font-weight: 800; color: #9aa3af; letter-spacing: 1.0px;"
        )
        page_lay.addWidget(lifetime_title)

        lifetime_row = QHBoxLayout()
        lifetime_row.setSpacing(10)
        self.card_lt_materials = KPICard("KALICI MATERYAL", "0", "Toplanan toplam (kalıcı)", "#56c8f5", parent=content)
        self.card_lt_gold = KPICard("KALICI ALTIN / KİNAH", "0", "Kazanılan altın / kinah (kalıcı)", "#f5b50a", parent=content)
        self.card_lt_kills = KPICard("KALICI CANAVAR", "0", "Öldürülen canavar (kalıcı)", "#fb4d6d", parent=content)
        lifetime_row.addWidget(self.card_lt_materials)
        lifetime_row.addWidget(self.card_lt_gold)
        lifetime_row.addWidget(self.card_lt_kills)
        page_lay.addLayout(lifetime_row)

        # 4. SEÇİLİ ARALIK İSTATİSTİKLERİ
        self.range_sec_title = QLabel("BUGÜN İSTATİSTİKLERİ (SEÇİLİ ARALIK)", content)
        self.range_sec_title.setStyleSheet(
            "font-size: 11px; font-weight: 800; color: #9aa3af; letter-spacing: 1.0px;"
        )
        page_lay.addWidget(self.range_sec_title)

        kpi_row1 = QHBoxLayout()
        kpi_row1.setSpacing(10)
        self.card_gathered = KPICard("TOPLANAN", "0", "Ort. Eşleşme Güveni: %0.0", "#56c8f5", parent=content)
        self.card_rate = KPICard(
            "SAATLİK TOPLAMA HIZI",
            "0.0",
            "Hedef 40/sa · %0 · aktif saat başına",
            "#34d399",
            with_progress=True,
            parent=content,
        )
        self.card_success = KPICard("BAŞARI ORANI", "%100.0", "0 başarısız deneme", "#fb923c", parent=content)
        self.card_walk = KPICard("ORT. YÜRÜME SÜRESİ", "0.00sn", "Temizlenen savaş: 0", "#a78bfa", parent=content)
        kpi_row1.addWidget(self.card_gathered)
        kpi_row1.addWidget(self.card_rate)
        kpi_row1.addWidget(self.card_success)
        kpi_row1.addWidget(self.card_walk)
        page_lay.addLayout(kpi_row1)

        kpi_row2 = QHBoxLayout()
        kpi_row2.setSpacing(10)
        self.card_kinah_total = KPICard("KİNAH / ALTIN", "0", "Seçili aralıkta kazanılan", "#f5b50a", compact=True, parent=content)
        self.card_kinah_rate = KPICard("SAATLİK KİNAH HIZI", "0/sa", "Aktif saat başına kinah", "#34d399", compact=True, parent=content)
        kpi_row2.addWidget(self.card_kinah_total, stretch=1)
        kpi_row2.addWidget(self.card_kinah_rate, stretch=1)
        kpi_row2.addStretch(2)
        page_lay.addLayout(kpi_row2)

        # 5. INSIGHTS STRIP (.insights)
        insights_row = QHBoxLayout()
        insights_row.setSpacing(10)
        self.ins_rota = _InsightCard("EN VERİMLİ ROTA", "—", "—", "#6b7280", parent=content)
        self.ins_kinah = _InsightCard("EN ÇOK KİNAH GETİREN", "—", "—", "#6b7280", parent=content)
        self.ins_dikkat = _InsightCard("DİKKAT", "—", "—", "#fb923c", parent=content)
        insights_row.addWidget(self.ins_rota, stretch=1)
        insights_row.addWidget(self.ins_kinah, stretch=1)
        insights_row.addWidget(self.ins_dikkat, stretch=1)
        page_lay.addLayout(insights_row)

        # 6. MIDDLE ROW (.midrow): Chart Card + Right Detail Inspector (302px)
        mid_row = QHBoxLayout()
        mid_row.setSpacing(10)

        chart_card = QFrame(content)
        chart_card.setObjectName("StatsChartCard")
        chart_lay = QVBoxLayout(chart_card)
        chart_lay.setContentsMargins(14, 11, 14, 10)
        chart_lay.setSpacing(6)

        chart_head = QHBoxLayout()
        self.chart_title_lbl = QLabel(
            "VERİM VE SAATLİK TOPLAMA HIZI (BUGÜN · 12 DİLİM) — DİLİME TIKLAYIN",
            chart_card,
        )
        self.chart_title_lbl.setStyleSheet(
            "color: #eef1f5; font-size: 12px; font-weight: 800; letter-spacing: 0.4px;"
        )
        # Long range names (e.g. SON 7 GÜN) must wrap, never clip, at 12px.
        self.chart_title_lbl.setWordWrap(True)
        chart_head.addWidget(self.chart_title_lbl)
        chart_head.addStretch(1)

        leg_lbl = QLabel(
            '<span style="color:#56c8f5;">■</span> Toplanan Materyal &nbsp;&nbsp;&nbsp; '
            '<span style="color:#34d399;">━</span> Saatlik Hız',
            chart_card,
        )
        leg_lbl.setTextFormat(Qt.TextFormat.RichText)
        leg_lbl.setStyleSheet("color: #9aa3af; font-size: 12px;")
        chart_head.addWidget(leg_lbl)
        chart_lay.addLayout(chart_head)

        self.chart_widget = LiveThroughputChart(chart_card)
        self.chart_widget.slice_clicked.connect(self._on_chart_slice_clicked)
        chart_lay.addWidget(self.chart_widget, stretch=1)
        mid_row.addWidget(chart_card, stretch=1)

        # Right Detail Inspector Card (302px wide)
        self.detail_card = QFrame(content)
        self.detail_card.setObjectName("StatsDetailCard")
        self.detail_card.setFixedWidth(320)
        det_lay = QVBoxLayout(self.detail_card)
        det_lay.setContentsMargins(16, 14, 16, 14)
        det_lay.setSpacing(8)

        self.detail_kicker_lbl = QLabel("ARALIK ÖZETİ", self.detail_card)
        self.detail_kicker_lbl.setStyleSheet(
            "color: #9aa3af; font-size: 11px; font-weight: 800; letter-spacing: 1.0px;"
        )
        self.detail_title_lbl = QLabel("Bugün Özeti", self.detail_card)
        self.detail_title_lbl.setStyleSheet("color: #eef1f5; font-size: 15px; font-weight: 800;")
        self.detail_sub_lbl = QLabel("0 kayıt · tüm bölgeler · tüm rotalar", self.detail_card)
        self.detail_sub_lbl.setStyleSheet("color: #9aa3af; font-size: 12px;")
        det_lay.addWidget(self.detail_kicker_lbl)
        det_lay.addWidget(self.detail_title_lbl)
        det_lay.addWidget(self.detail_sub_lbl)

        self.detail_kv_labels: List[Tuple[QLabel, QLabel]] = []
        for k_txt in (
            "Toplam deneme",
            "Farklı kaynak türü",
            "Farklı rota",
            "Savaş / temizlenen",
            "Öldürülen canavar",
            "Başarısız deneme",
        ):
            r_lay = QHBoxLayout()
            r_lay.setContentsMargins(0, 2, 0, 2)
            kl = QLabel(k_txt, self.detail_card)
            kl.setStyleSheet("color: #9aa3af; font-size: 12px;")
            vl = QLabel("0", self.detail_card)
            vl.setStyleSheet(
                'color: #eef1f5; font-family: Consolas, "Cascadia Mono", monospace; font-size: 12px; font-weight: 700;'
            )
            r_lay.addWidget(kl)
            r_lay.addStretch(1)
            r_lay.addWidget(vl)
            self.detail_kv_labels.append((kl, vl))
            det_lay.addLayout(r_lay)

        self.detail_events_title = QLabel("SON OLAYLAR", self.detail_card)
        self.detail_events_title.setStyleSheet(
            "color: #6b7280; font-size: 10px; font-weight: 800; letter-spacing: 1.0px; margin-top: 6px;"
        )
        det_lay.addWidget(self.detail_events_title)

        self.detail_events_list = QListWidget(self.detail_card)
        self.detail_events_list.setMaximumHeight(118)
        self.detail_events_list.setStyleSheet(
            "QListWidget { background-color: #0d1015; border: 1px solid #1e2530; border-radius: 6px; "
            "color: #eef1f5; font-size: 11px; outline: none; }"
            "QListWidget::item { padding: 4px 6px; border-bottom: 1px solid #1e2530; }"
            "QListWidget::item:selected { background-color: rgba(245, 181, 10, 0.14); color: #eef1f5; }"
            "QListWidget::item:disabled { color: #6b7280; }"
        )
        self.detail_events_list.itemClicked.connect(self._on_event_selected)
        det_lay.addWidget(self.detail_events_list)

        det_lay.addStretch(1)

        self.btn_group_csv = QPushButton("Grubu CSV İndir", self.detail_card)
        self.btn_group_csv.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_group_csv.setVisible(False)
        self.btn_group_csv.setToolTip("Seçili kaynak grubunun olaylarını CSV olarak kaydeder")
        self.btn_group_csv.clicked.connect(self.on_export_group_csv)
        det_lay.addWidget(self.btn_group_csv)

        self.btn_clear_selection = QPushButton("Seçimi Temizle", self.detail_card)
        self.btn_clear_selection.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear_selection.setVisible(False)
        self.btn_clear_selection.clicked.connect(self._clear_group_selection)
        det_lay.addWidget(self.btn_clear_selection)

        mid_row.addWidget(self.detail_card)
        page_lay.addLayout(mid_row)

        # 7. DISTRIBUTION TABLE (.tablecard)
        table_head_row = QHBoxLayout()
        table_title = QLabel("BÖLGE/NOKTA VERİMLİLİĞİ VE KAYNAK DAĞILIMI", content)
        table_title.setStyleSheet(
            "font-size: 11px; font-weight: 800; color: #9aa3af; letter-spacing: 1.0px;"
        )
        table_head_row.addWidget(table_title)
        table_head_row.addStretch(1)
        table_hint = QLabel("Başlığa tıklayarak sıralayın · satıra tıklayınca detay açılır", content)
        table_hint.setStyleSheet("color: #6b7280; font-size: 12px;")
        table_head_row.addWidget(table_hint)
        page_lay.addLayout(table_head_row)

        table_card = QFrame(content)
        table_card.setObjectName("StatsTableCard")
        tc_lay = QVBoxLayout(table_card)
        tc_lay.setContentsMargins(0, 0, 0, 0)
        tc_lay.setSpacing(0)

        self.table = QTableWidget(0, 9, table_card)
        self.table.setHorizontalHeaderLabels(
            [
                "KAYNAK TÜRÜ",
                "BÖLGE",
                "ROTA",
                "ADET",
                "PAY %",
                "ORT. YÜRÜME (sn)",
                "KİNAH",
                "ORT. GÜVEN",
                "SON",
            ]
        )
        self.table.setMinimumHeight(180)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)

        header = self.table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(4, 125)
            header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(8, QHeaderView.ResizeMode.ResizeToContents)
            header.sectionClicked.connect(self._on_header_clicked)
        tc_lay.addWidget(self.table)
        page_lay.addWidget(table_card, stretch=1)

        # 8. CANLI GÜNLÜK (.logcard) — bot logger'larından gerçek kayıtlar.
        log_card = QFrame(content)
        log_card.setObjectName("StatsLogCard")
        log_card.setStyleSheet(
            "QFrame#StatsLogCard { background-color: #05070a; border: 1px solid #1e2530; border-radius: 9px; }"
        )
        log_lay = QVBoxLayout(log_card)
        log_lay.setContentsMargins(0, 0, 0, 0)
        log_lay.setSpacing(0)

        log_head = QHBoxLayout()
        log_head.setContentsMargins(12, 6, 12, 6)
        log_head.setSpacing(8)
        log_title = QLabel("CANLI GÜNLÜK", log_card)
        log_title.setStyleSheet("color: #eef1f5; font-size: 11px; font-weight: 800; letter-spacing: 0.6px;")
        log_head.addWidget(log_title)
        log_head.addStretch(1)

        self._log_filter_group = QButtonGroup(log_card)
        self._log_filter_group.setExclusive(True)
        self._log_filter_buttons: Dict[str, QPushButton] = {}
        for f_key, f_label in (("all", "Tümü"), ("INFO", "INFO"), ("WARNING", "WARNING")):
            fb = QPushButton(f_label, log_card)
            fb.setCheckable(True)
            fb.setChecked(f_key == "all")
            fb.setCursor(Qt.CursorShape.PointingHandCursor)
            fb.setStyleSheet(
                "QPushButton { background: transparent; border: 1px solid #1e2530; color: #9aa3af; "
                "border-radius: 99px; padding: 3px 10px; font-size: 11px; font-weight: 700; }"
                "QPushButton:checked { background: #ffffff; border-color: #ffffff; color: #000000; }"
            )
            fb.clicked.connect(lambda _c=False, fk=f_key: self._on_log_filter_selected(fk))
            self._log_filter_group.addButton(fb)
            self._log_filter_buttons[f_key] = fb
            log_head.addWidget(fb)

        self.btn_clear_log = QPushButton("Temizle", log_card)
        self.btn_clear_log.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear_log.setStyleSheet(
            "QPushButton { background: transparent; border: 1px solid #1e2530; color: #fca5a5; "
            "border-radius: 99px; padding: 3px 10px; font-size: 11px; font-weight: 700; }"
        )
        self.btn_clear_log.clicked.connect(self._on_clear_log)
        log_head.addWidget(self.btn_clear_log)
        log_lay.addLayout(log_head)

        self.log_view = QPlainTextEdit(log_card)
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(72)
        self.log_view.setMaximumBlockCount(300)
        self.log_view.setStyleSheet(
            "QPlainTextEdit { background-color: #05070a; color: #c3ccd9; border: none; "
            'font-family: Consolas, "Cascadia Mono", monospace; font-size: 11px; }'
        )
        log_lay.addWidget(self.log_view)
        page_lay.addWidget(log_card)

        scroll.setWidget(content)
        root_layout.addWidget(scroll, stretch=1)

        # 9. BOTTOM ACTIONS (.actions)
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(6, 0, 6, 0)
        btn_layout.setSpacing(8)

        self.btn_export_csv = QPushButton("CSV Dışa Aktar", self)
        self.btn_export_csv.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export_csv.setStyleSheet(
            "QPushButton { background-color: #1d7fd1; border-color: #3694e3; color: #ffffff; padding: 10px 16px; } "
            "QPushButton:hover { background-color: #2589de; }"
        )
        self.btn_export_csv.clicked.connect(self.on_export_csv)

        self.btn_save_json = QPushButton("JSON Rapor Kaydet", self)
        self.btn_save_json.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save_json.setStyleSheet(
            "QPushButton { background-color: #1f9d63; border-color: #2cb676; color: #ffffff; padding: 10px 16px; } "
            "QPushButton:hover { background-color: #24ad6e; }"
        )
        self.btn_save_json.clicked.connect(self.on_save_json)

        self.btn_reset = QPushButton("Oturumu Sıfırla", self)
        self.btn_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset.setStyleSheet(
            "QPushButton { background-color: #d97a06; border-color: #f08c0e; color: #ffffff; padding: 10px 16px; } "
            "QPushButton:hover { background-color: #e58309; }"
        )
        self.btn_reset.clicked.connect(self.on_reset_stats)

        self.btn_reset_lifetime = QPushButton("Tüm Veriyi Sıfırla", self)
        self.btn_reset_lifetime.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset_lifetime.setStyleSheet(
            "QPushButton { background-color: #d81b4c; border-color: #ec3564; color: #ffffff; padding: 10px 16px; } "
            "QPushButton:hover { background-color: #e42254; }"
        )
        self.btn_reset_lifetime.setToolTip("Ömür boyu istatistikleri siler — yalnızca açık onayla")
        self.btn_reset_lifetime.clicked.connect(self.on_reset_lifetime_stats)

        self.btn_close = QPushButton("Kapat", self)
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_close.setStyleSheet(
            "QPushButton { background-color: #141923; border-color: #1e2530; color: #eef1f5; padding: 10px 18px; } "
            "QPushButton:hover { background-color: #1b2230; }"
        )
        self.btn_close.clicked.connect(self.close)

        btn_layout.addWidget(self.btn_export_csv)
        btn_layout.addWidget(self.btn_save_json)
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addWidget(self.btn_reset_lifetime)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.btn_close)

        root_layout.addLayout(btn_layout)

    def _on_speed_changed(self, _index: int = 0) -> None:
        ms = int(self.speed_combo.currentData() or 1000)
        self._refresh_timer.setInterval(ms)

    def _toggle_live_timer(self) -> None:
        if self._refresh_timer.isActive():
            self._refresh_timer.stop()
            self.btn_live.setText("Canlı Akışı Başlat")
        else:
            self._refresh_timer.start()
            self.btn_live.setText("Canlı Akışı Duraklat")
            self.refresh_metrics()

    # -- Canlı Günlük ---------------------------------------------------
    def _setup_live_log(self) -> None:
        """Bot logger'larına handler bağlar + güncel oturum log dosyası kuyruğunu yükler.

        Gösterilen her satır ya botun gerçek log kaydıdır ya da dialog'un
        kendi eylem kaydıdır (açılış, filtre, dışa aktarma...). Sahte
        hazır satır yoktur.
        """
        try:
            self.log_record_received.connect(self._on_log_record)
            self._log_handler = _QtLogHandler(weakref.ref(self))
            for name in _LIVE_LOG_SOURCES:
                lg = logging.getLogger(name)
                if lg.level == logging.NOTSET:
                    lg.setLevel(logging.INFO)
                if self._log_handler not in lg.handlers:
                    lg.addHandler(self._log_handler)
        except Exception as exc:
            logger.debug(f"Canlı günlük handler bağlanamadı: {exc}")
        self._tail_bot_log()
        self._append_log_record("INFO", "İstatistik sayfası açıldı — kayıtlı olaylar yüklendi")

    def _tail_bot_log(self, max_lines: int = 40) -> None:
        """Güncel oturum log dosyasının son satırlarını Canlı Günlük'e yükler (dosya yoksa atlanır)."""
        try:
            from core.session_log import resolve_current_log_path

            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            resolved = resolve_current_log_path(os.path.join(repo_root, "logs"))
            if resolved is None:
                return
            path = os.fspath(resolved)
            if not os.path.isfile(path):
                return
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()[-max(1, int(max_lines)):]
            for raw in lines:
                line = raw.rstrip("\n")
                if not line.strip():
                    continue
                m = _LOG_LINE_RE.match(line)
                if m:
                    self._append_log_record(
                        m.group("level"), m.group("msg"), time_str=m.group("time")
                    )
                else:
                    self._append_log_record("INFO", line)
        except Exception as exc:
            # D-3 (dilim-29): eskiden yalnızca logger.debug — kullanıcı
            # Canlı Günlük'ü boş görüyor, nedenini bulamıyordu (debug
            # görünmüyor). Artık warning + panelde görünür kayıt.
            logger.warning(f"günlük kuyruğu okunamadı ({os.path.basename(path)}): {exc}")
            try:
                self._append_log_record(
                    "WARNING",
                    f"günlük dosyası okunamadı ({exc}) — Canlı Günlük yalnızca canlı kayıtları gösterecek.",
                )
            except Exception:
                pass

    def _log_visible(self, level: str) -> bool:
        if self._log_filter == "all":
            return True
        if self._log_filter == "INFO":
            return level == "INFO"
        # WARNING filtresi: WARNING ve üzeri (ERROR/CRITICAL) gösterilir.
        return level in ("WARNING", "ERROR", "CRITICAL")

    def _append_log_record(self, level: str, msg: str, time_str: Optional[str] = None) -> None:
        lvl = str(level or "INFO").upper()
        ts = time_str or datetime.now().strftime("%H:%M:%S")
        self._log_records.append((ts, lvl, str(msg)))
        if self._log_visible(lvl):
            try:
                self.log_view.appendPlainText(f"[{ts}] [{lvl}] {msg}")
            except Exception:
                pass

    def _on_log_record(self, level: str, msg: str) -> None:
        self._append_log_record(level, msg)

    def _on_log_filter_selected(self, filter_key: str) -> None:
        self._log_filter = str(filter_key)
        try:
            self.log_view.clear()
            for ts, lvl, msg in self._log_records:
                if self._log_visible(lvl):
                    self.log_view.appendPlainText(f"[{ts}] [{lvl}] {msg}")
        except Exception as exc:
            logger.debug(f"log filtre yenileme hatası: {exc}")

    def _on_clear_log(self) -> None:
        self._log_records.clear()
        try:
            self.log_view.clear()
        except Exception:
            pass
        self._append_log_record("INFO", "Günlük temizlendi.")

    def _on_range_selected(self, range_key: str) -> None:
        self._active_range = str(range_key)
        self._selected_slice = None
        self._selected_event = None
        self.chart_widget.set_selected_slice(None)
        self.slice_chip_btn.setVisible(False)
        self.refresh_metrics()

    def _on_filter_changed(self, _index: int = 0) -> None:
        if self._updating_filters:
            return
        self._selected_event = None
        self.refresh_metrics()

    def _on_chart_slice_clicked(self, idx: int) -> None:
        if idx < 0 or self._selected_slice == idx:
            self._selected_slice = None
            self.chart_widget.set_selected_slice(None)
            self.slice_chip_btn.setVisible(False)
        else:
            self._selected_slice = idx
            self.chart_widget.set_selected_slice(idx)
            buckets = self.chart_widget.buckets
            lbl = f"#{idx + 1}"
            if 0 <= idx < len(buckets) and isinstance(buckets[idx], dict):
                try:
                    b_from = datetime.fromtimestamp(float(buckets[idx].get("start_ts", 0) or 0)).strftime("%H:%M")
                    b_to = datetime.fromtimestamp(float(buckets[idx].get("end_ts", 0) or 0)).strftime("%H:%M")
                    lbl = f"{b_from}–{b_to}"
                except (TypeError, ValueError, OSError):
                    lbl = str(buckets[idx].get("label", f"#{idx + 1}"))
            self.slice_chip_btn.setText(f"Dilim: {lbl} ×")
            self.slice_chip_btn.setVisible(True)
        self._selected_event = None
        self.refresh_metrics()

    def _on_header_clicked(self, col_index: int) -> None:
        col_map = {
            0: "resource_type",
            1: "region",
            2: "route",
            3: "count",
            4: "share_pct",
            5: "avg_walk_sec",
            6: "kinah_value",
            7: "avg_conf",
        }
        key = col_map.get(col_index, "count")
        if self._sort_col == key:
            self._sort_dir *= -1
        else:
            self._sort_col = key
            self._sort_dir = 1 if key in ("resource_type", "region", "route") else -1
        self._populate_table(self._last_rows)

    def _on_table_selection_changed(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            self._selected_group_key = None
            self._selected_event = None
            self._update_detail_inspector()
            return
        row_idx = selected[0].row()
        it = self.table.item(row_idx, 0)
        if it is not None:
            self._selected_group_key = str(it.data(Qt.ItemDataRole.UserRole) or "")
        self._selected_event = None
        self._update_detail_inspector()

    def _clear_group_selection(self) -> None:
        self._selected_group_key = None
        self._selected_event = None
        self.table.clearSelection()
        self._update_detail_inspector()

    # -- Detay paneli: son olaylar -------------------------------------
    def _recent_events(self, limit: int = 6) -> List[Tuple[str, Dict[str, Any]]]:
        """Seçili aralık + dilim + bölge/rota + grup filtrelerine uyan son olaylar.

        Kalıcı ("all") aralıkta olay kaydı tutulmaz — boş liste döner.
        Olay başına kinah telemetride tutulmaz; kinah yalnızca broker
        fiyatından tahmin edilir (detay panelinde "tahmini" diye yazılır).
        """
        try:
            tel = getattr(self, "telemetry", None)
            if tel is None:
                return []
            start, now, mode = self._range_window()
            if mode == "all":
                return []
            metrics = self._last_metrics if isinstance(self._last_metrics, dict) else {}
            slice_start = float(metrics.get("slice_start_ts", start) or start)
            slice_end = float(metrics.get("slice_end_ts", now) or now)
            reg_sel = str(self.region_filter.currentData() or "ALL")
            rte_sel = str(self.route_filter.currentData() or "ALL")
            group_key = str(self._selected_group_key or "")

            def _in_window(ev: Any) -> bool:
                if not isinstance(ev, dict):
                    return False
                try:
                    ts = float(ev.get("timestamp", 0.0) or 0.0)
                except (TypeError, ValueError):
                    return False
                return slice_start <= ts <= slice_end

            def _group_match(ev: Dict[str, Any]) -> bool:
                if not group_key:
                    return True
                key = (
                    f"{ev.get('resource_type', '')}|{ev.get('region', '')}|"
                    f"{ev.get('route', '')}"
                )
                return key == group_key

            found: List[Tuple[float, str, Dict[str, Any]]] = []
            for ev in list(getattr(tel, "gather_events", None) or []):
                if not isinstance(ev, dict) or not _in_window(ev):
                    continue
                if reg_sel != "ALL" and str(ev.get("region", "")) != reg_sel:
                    continue
                if rte_sel != "ALL" and str(ev.get("route", "")) != rte_sel:
                    continue
                if not _group_match(ev):
                    continue
                found.append((float(ev.get("timestamp", 0.0) or 0.0), "gather", ev))
            # Başarısızlık ve savaş olayları bölge/rota taşımaz: yalnızca
            # filtreler "Tümü" iken ve grup seçili değilken listelenir.
            if reg_sel == "ALL" and rte_sel == "ALL" and not group_key:
                for ev in list(getattr(tel, "failure_events", None) or []):
                    if isinstance(ev, dict) and _in_window(ev):
                        found.append((float(ev.get("timestamp", 0.0) or 0.0), "fail", ev))
                for ev in list(getattr(tel, "combat_events", None) or []):
                    if isinstance(ev, dict) and _in_window(ev):
                        found.append((float(ev.get("timestamp", 0.0) or 0.0), "combat", ev))
            found.sort(key=lambda t: t[0], reverse=True)
            return [(kind, ev) for _ts, kind, ev in found[: max(0, int(limit))]]
        except Exception as exc:
            logger.debug(f"_recent_events hatası: {exc}")
            return []

    def _event_kinah_text(self, ev: Dict[str, Any]) -> str:
        """Olayın tahmini kinah değeri — broker fiyatı bilinmiyorsa 'fiyat yok'."""
        try:
            price = self._known_unit_price(str(ev.get("resource_type", "")))
            if price is None:
                return "fiyat yok"
            items = int(ev.get("items_gathered", 1) or 1)
            return f"{format_kinah_tr(price * items)} (tahmini)"
        except (TypeError, ValueError):
            return "fiyat yok"

    def _on_event_selected(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], dict):
            self._selected_event = (str(data[0]), dict(data[1]))
        else:
            self._selected_event = None
        self._update_detail_inspector()

    def _known_unit_price(self, res_name: str) -> Optional[int]:
        """F-0547: unit Kinah price only when the broker actually knows it.
        TradeBrokerPriceScanner.get_price() returns None for unknown materials
        when no explicit default is given (P0-7), so 'known' is checked
        against the price table (exact / case-insensitive match only —
        substring matching was removed: e.g. 'Odyle' must not price
        'Pure Odyle'). Returns None when the price is unknown — callers
        must not invent one.
        """
        name = str(res_name or "").strip()
        if not name:
            return None
        econ = getattr(self, "economy_calculator", None)
        scanner = getattr(econ, "price_scanner", econ) if econ is not None else None
        prices = getattr(scanner, "prices", None) if scanner is not None else None
        if isinstance(prices, dict):
            # S110 (v19/P3-8): fiyat tablosundaki sayısal olmayan
            # değer (`"1.5k"`, `"abc"`) `int()`'te patlıyor, try
            # yalnızca `_row_kinah`'ın çarpım satırını sarıyordu —
            # fiyat okuma try DIŞINDAYDI, yenileme çöküyordu.
            # Bilinmeyen/bozuk fiyat = fiyat bilinmiyor (None).
            if name in prices:
                try:
                    return int(prices[name])
                except (TypeError, ValueError):
                    return None
            low = name.lower()
            for k, v in prices.items():
                if str(k).lower() == low:
                    try:
                        return int(v)
                    except (TypeError, ValueError):
                        return None
            return None
        get_price = getattr(scanner, "get_price", None) if scanner is not None else None
        if callable(get_price):
            try:
                val = int(get_price(name, 0) or 0)
            except Exception:
                return None
            return val if val > 0 else None
        return None

    def _row_kinah(self, row: Dict[str, Any]) -> Optional[int]:
        """Known-price Kinah value of a breakdown row, or None."""
        if not isinstance(row, dict):
            return None
        price = self._known_unit_price(str(row.get("resource_type", "")))
        if price is None:
            return None
        try:
            return int(row.get("count", 0) or 0) * int(price)
        except (TypeError, ValueError):
            return None

    def _range_window(self) -> Tuple[float, float, str]:
        """F-0522: (start_ts, now, mode) for the active range filter."""
        now = time.time()
        key = str(getattr(self, "_active_range", "today") or "today")
        if key == "session":
            tel = getattr(self, "telemetry", None)
            start = float(getattr(tel, "session_start_ts", now) or now)
            return start, now, "session"
        if key == "7d":
            return now - 7 * 86400.0, now, "7d"
        if key == "all":
            return 0.0, now, "all"
        midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight.timestamp(), now, "today"

    @staticmethod
    def _build_buckets(
        events: List[Dict[str, Any]],
        failures: List[Dict[str, Any]],
        combats: List[Dict[str, Any]],
        start: float,
        now: float,
    ) -> List[Dict[str, Any]]:
        buckets: List[Dict[str, Any]] = []
        span = max(5.0, (max(now, start) - start) / 12.0)
        for i in range(12):
            b_start = start + i * span
            b_end = b_start + span
            buckets.append(
                {
                    "index": i,
                    "label": datetime.fromtimestamp(b_end).strftime("%H:%M:%S"),
                    "start_ts": b_start,
                    "end_ts": b_end,
                    "count": 0,
                    "failures": 0,
                    "combats": 0,
                    "rate_per_hour": 0.0,
                }
            )
        for ev in events:
            ts = float(ev.get("timestamp", 0.0) or 0.0)
            idx = int((ts - start) / span) if span > 0 else -1
            if 0 <= idx < 12:
                buckets[idx]["count"] += 1
        for ev in failures:
            ts = float(ev.get("timestamp", 0.0) or 0.0)
            idx = int((ts - start) / span) if span > 0 else -1
            if 0 <= idx < 12:
                buckets[idx]["failures"] += 1
        for ev in combats:
            ts = float(ev.get("timestamp", 0.0) or 0.0)
            idx = int((ts - start) / span) if span > 0 else -1
            if 0 <= idx < 12:
                buckets[idx]["combats"] += 1
        for b in buckets:
            b["rate_per_hour"] = round((b["count"] / span) * 3600.0, 2)
        return buckets

    @staticmethod
    def _merged_events(live: Any, history: Any, kind: str) -> List[Dict[str, Any]]:
        """Canlı + geçmiş oturum olaylarını birleştir (tekilleştir; canlı kazanır).

        Geçmiş olaylar `TelemetryEngine.load_history_events` ile ayrı
        listelerde tutulur; aynı olay birden çok snapshot'ta geçebilir.
        Anahtar: (zaman, düğüm, tür/sonuç) — canlı listedeki kopya
        geçmiştekiyle aynıysa bir kez sayılır.
        """
        merged: Dict[Any, Dict[str, Any]] = {}

        def _key(ev: Dict[str, Any]) -> Any:
            try:
                ts = round(float(ev.get("timestamp", 0.0) or 0.0), 3)
            except (TypeError, ValueError):
                ts = 0.0
            if kind == "gather":
                return ("g", ts, ev.get("node_id"), str(ev.get("resource_type", "")))
            if kind == "failure":
                return ("f", ts, ev.get("node_id"), str(ev.get("reason", "")))
            return ("c", ts, str(ev.get("outcome", "")), ev.get("duration_sec"))

        for ev in history or []:
            if isinstance(ev, dict):
                merged[_key(ev)] = ev
        for ev in live or []:
            if isinstance(ev, dict):
                merged[_key(ev)] = ev
        def _ts(ev: Dict[str, Any]) -> float:
            try:
                return float(ev.get("timestamp", 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        return sorted(merged.values(), key=_ts)

    def _metrics_for_range(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """F-0522/F-0617: recompute the displayed metrics for the active
        range (and the selected chart slice, if any) from raw telemetry
        events instead of always showing whole-session numbers."""
        base = dict(base) if isinstance(base, dict) else {}
        tel = getattr(self, "telemetry", None)
        start, now, mode = self._range_window()
        buckets = list(base.get("hourly_buckets") or [])
        slice_start, slice_end = start, now
        if self._selected_slice is not None and buckets:
            idx = int(self._selected_slice)
            if 0 <= idx < len(buckets) and isinstance(buckets[idx], dict):
                slice_start = float(buckets[idx].get("start_ts", start) or start)
                slice_end = float(buckets[idx].get("end_ts", now) or now)

        if mode == "all":
            try:
                snap = get_lifetime_stats().snapshot()
            except Exception:
                snap = {}
            mats = snap.get("materials_by_key") or {}
            if not isinstance(mats, dict):
                mats = {}
            gathered = int(snap.get("materials_total", 0) or 0)
            gold = int(snap.get("gold_earned", 0) or 0)
            kills = int(snap.get("monsters_killed", 0) or 0)
            rows: List[Dict[str, Any]] = []
            for res, qty in sorted(mats.items(), key=lambda kv: -int(kv[1] or 0)):
                try:
                    q = int(qty or 0)
                except (TypeError, ValueError):
                    continue
                if q <= 0:
                    continue
                # P1-6 (K-06): türsüz kalıcı kayıt "Orichalcum Ore"
                # sayılmaz — boş tür ayrı "Unknown" kovasına girer.
                res_name = (str(res).strip() or "Unknown")
                rows.append(
                    {
                        "resource_type": res_name,
                        "region": "Kalıcı",
                        "route": "Kalıcı",
                        "count": q,
                        "share_pct": round((q / max(1, gathered)) * 100.0, 1),
                        "avg_walk_sec": 0.0,
                        "avg_match_pct": 0.0,
                    }
                )
            out = dict(base)
            out.update(
                {
                    "range_mode": "all",
                    "session_duration_sec": float(base.get("session_duration_sec") or 0.0),
                    "gathered_total": gathered,
                    "failed_total": 0,
                    "combat_count": int(base.get("combat_count") or 0),
                    "gathers_per_hour": 0.0,
                    "avg_walk_sec": 0.0,
                    "avg_match_pct": 0.0,
                    # Lifetime history has no attempt log: success is unknown, not 100%.
                    "success_rate_pct": None,
                    "kills_in_range": kills,
                    "kills_label": "Öldürülen canavar (kalıcı)",
                    "by_resource": dict(mats),
                    "by_route": {},
                    "by_region": {},
                    "failure_reasons": {},
                    "detailed_breakdown": rows,
                    "lifetime_gold": gold,
                    "lifetime_kills": kills,
                }
            )
            return out

        def _in_window(ev: Any) -> bool:
            if not isinstance(ev, dict):
                return False
            try:
                ts = float(ev.get("timestamp", 0.0) or 0.0)
            except (TypeError, ValueError):
                return False
            return slice_start <= ts <= slice_end

        events = [e for e in self._merged_events(getattr(tel, "gather_events", None), getattr(tel, "history_gather_events", None), "gather") if _in_window(e)]
        failures = [e for e in self._merged_events(getattr(tel, "failure_events", None), getattr(tel, "history_failure_events", None), "failure") if _in_window(e)]
        combats = [e for e in self._merged_events(getattr(tel, "combat_events", None), getattr(tel, "history_combat_events", None), "combat") if _in_window(e)]
        gathered = len(events)
        failed = len(failures)
        # P1-6 (K-06): oturum aralığı + dilim seçili değilken toplamlar
        # kaplı olay listesinden değil telemetrinin kümülatif
        # sayaçlarından okunur (5000 gather sonrası liste başından
        # atar; toplam takılmaz). Kırılım satırları/paylar ise tutulan
        # pencereye göre hesaplanmaya devam eder (aşağıda window_total).
        slice_selected = self._selected_slice is not None and bool(buckets)
        window_total = max(1, len(events))
        if mode == "session" and not slice_selected:
            gathered = int(base.get("gathered_total") or gathered)
            failed = int(base.get("failed_total") or failed)
            combats_n = int(base.get("combat_count") or len(combats))
        else:
            combats_n = len(combats)
        total_attempts = gathered + failed
        # v20-istatistik (R1): aralık içi öldürme — yalnızca temizlenen
        # savaşlar sayılır (ölüm/kaçış değil); kalıcı sayaç buraya girmez.
        kills_in_range = sum(
            1
            for c in combats
            if str(c.get("outcome", "") or "").upper()
            in ("CLEARED", "WIN", "VICTORY", "KILLED", "SUCCESS")
        )
        elapsed_h = max(1.0 / 3600.0, (slice_end - slice_start) / 3600.0)
        # P1-6: ortalamalar tutulan pencere olaylarına bölünür
        # (toplamlar kümülatif olabilir; payda pencere sayısıdır).
        if events:
            avg_walk = round(sum(float(e.get("walk_duration_sec", 0.0) or 0.0) for e in events) / len(events), 2)
            avg_match = round(sum(int(e.get("match_pct", 0) or 0) for e in events) / len(events), 1)
        else:
            avg_walk, avg_match = 0.0, 0.0
        by_resource: Dict[str, int] = {}
        by_route: Dict[str, int] = {}
        by_region: Dict[str, int] = {}
        failure_reasons: Dict[str, int] = {}
        groups: Dict[str, Dict[str, Any]] = {}
        for e in events:
            # P1-6 (K-06): türsüz olay "Orichalcum Ore" sayılmaz —
            # türü yoksa/boşsa ayrı "Unknown" kovasına girer.
            res = (str(e.get("resource_type") or "Unknown").strip() or "Unknown")
            reg = str(e.get("region", "Unknown"))
            rte = str(e.get("route", "Default Route"))
            by_resource[res] = by_resource.get(res, 0) + 1
            by_route[rte] = by_route.get(rte, 0) + 1
            by_region[reg] = by_region.get(reg, 0) + 1
            key = f"{res}||{reg}||{rte}"
            g = groups.setdefault(
                key,
                {"resource_type": res, "region": reg, "route": rte, "count": 0,
                 "total_walk_sec": 0.0, "total_match_pct": 0},
            )
            g["count"] += 1
            try:
                g["total_walk_sec"] += float(e.get("walk_duration_sec", 0.0) or 0.0)
                g["total_match_pct"] += int(e.get("match_pct", 0) or 0)
            except (TypeError, ValueError):
                pass
        for f in failures:
            reason = str(f.get("reason", "timeout") or "timeout")
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        rows = []
        for g in groups.values():
            cnt = max(1, int(g["count"]))
            rows.append(
                {
                    "resource_type": g["resource_type"],
                    "region": g["region"],
                    "route": g["route"],
                    "count": int(g["count"]),
                    # P1-6: pay, satırların hesaplandığı tutulan
                    # pencereye bölünür (kümülatif toplama değil).
                    "share_pct": round((int(g["count"]) / window_total) * 100.0, 1),
                    "avg_walk_sec": round(float(g["total_walk_sec"]) / cnt, 2),
                    "avg_match_pct": round(float(g["total_match_pct"]) / cnt, 1),
                }
            )
        rows.sort(key=lambda r: r["count"], reverse=True)
        out = dict(base)
        out.update(
            {
                "range_mode": mode,
                "range_start_ts": start,
                "range_end_ts": now,
                "slice_start_ts": slice_start,
                "slice_end_ts": slice_end,
                "gathered_total": gathered,
                "failed_total": failed,
                "combat_count": combats_n,
                "kills_in_range": kills_in_range,
                "kills_label": "Öldürülen canavar",
                "combat_duration_sec": round(
                    sum(float(c.get("duration_sec", 0.0) or 0.0) for c in combats), 2
                ),
                "gathers_per_hour": round((gathered / elapsed_h), 2),
                "avg_walk_sec": avg_walk,
                "avg_match_pct": avg_match,
                # F-0708: no attempts yet -> unknown (None), never 100% or 0%.
                "success_rate_pct": (
                    round((gathered / total_attempts) * 100.0, 2) if total_attempts > 0 else None
                ),
                "by_resource": by_resource,
                "by_route": by_route,
                "by_region": by_region,
                "failure_reasons": failure_reasons,
                "detailed_breakdown": rows,
                "hourly_buckets": self._build_buckets(events, failures, combats, start, now),
            }
        )
        return out

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = max(0, int(seconds))
        hrs = total // 3600
        mins = (total % 3600) // 60
        secs = total % 60
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"

    def _refresh_lifetime_cards(self) -> None:
        try:
            snap = get_lifetime_stats().snapshot()
            mats_total = int(snap.get("materials_total", 0) or 0)
            gold = int(snap.get("gold_earned", 0) or 0)
            kills = int(snap.get("monsters_killed", 0) or 0)
            mats_by = snap.get("materials_by_key") or {}
            kinds = len(mats_by) if isinstance(mats_by, dict) else 0
            lu = snap.get("last_updated") or "—"
            self.card_lt_materials.update_card(f"{mats_total}", f"{kinds} tür · son: {lu}")
            self.card_lt_gold.update_card(format_kinah_tr(gold), "Kazanılan altın / kinah (kalıcı)")
            self.card_lt_kills.update_card(f"{kills}", "Öldürülen canavar (kalıcı)")
            self.lifetime_info_lbl.setText(
                f"Kalıcı: Materyal {mats_total} · Altın {format_kinah_tr(gold)} · Canavar {kills}"
            )
        except Exception as exc:
            logger.warning("lifetime cards refresh failed: %s", exc)
            try:
                self.card_lt_materials.update_card("—", "yüklenemedi")
                self.card_lt_gold.update_card("—", "yüklenemedi")
                self.card_lt_kills.update_card("—", "yüklenemedi")
                self.lifetime_info_lbl.setText("Kalıcı: yüklenemedi")
            except Exception:
                pass

    def _sync_filter_combos(self, rows: List[Dict[str, Any]]) -> None:
        self._updating_filters = True
        try:
            cur_reg = str(self.region_filter.currentData() or "ALL")
            cur_rte = str(self.route_filter.currentData() or "ALL")
            regions = sorted({str(r.get("region", "")).strip() for r in rows if str(r.get("region", "")).strip()})
            routes = sorted({str(r.get("route", "")).strip() for r in rows if str(r.get("route", "")).strip()})

            self.region_filter.clear()
            self.region_filter.addItem("Tümü", "ALL")
            for reg in regions:
                self.region_filter.addItem(reg, reg)
            idx_r = self.region_filter.findData(cur_reg)
            self.region_filter.setCurrentIndex(max(0, idx_r))

            self.route_filter.clear()
            self.route_filter.addItem("Tümü", "ALL")
            for rte in routes:
                self.route_filter.addItem(rte, rte)
            idx_rt = self.route_filter.findData(cur_rte)
            self.route_filter.setCurrentIndex(max(0, idx_rt))
        finally:
            self._updating_filters = False

    def refresh_metrics(self) -> None:
        """TelemetryEngine ve SessionProfitSimulator üzerinden canlı verileri çekip tüm panelleri günceller."""
        try:
            self._refresh_lifetime_cards()

            raw_metrics = self.telemetry.get_live_metrics()
            base_metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
            metrics = self._metrics_for_range(base_metrics)
            self._last_metrics = metrics
            range_mode = str(metrics.get("range_mode") or self._active_range)
            # Session label always shows the live session length, even when
            # the range filter shows today / 7 days / lifetime numbers.
            duration_sec = float(base_metrics.get("session_duration_sec") or 0.0)
            duration_str = self._format_duration(duration_sec)
            combats = int(metrics.get("combat_count") or 0)
            failures = int(metrics.get("failed_total") or 0)
            gathered = int(metrics.get("gathered_total") or 0)
            rate_hr = float(metrics.get("gathers_per_hour") or 0.0)
            success_raw = metrics.get("success_rate_pct")
            success_pct = float(success_raw) if success_raw is not None else None
            avg_walk = float(metrics.get("avg_walk_sec") or 0.0)
            avg_match = float(metrics.get("avg_match_pct") or 0.0)

            all_rows = [
                r for r in (metrics.get("detailed_breakdown", []) or []) if isinstance(r, dict)
            ]
            self._sync_filter_combos(all_rows)

            reg_sel = str(self.region_filter.currentData() or "ALL")
            rte_sel = str(self.route_filter.currentData() or "ALL")
            rows = [
                r
                for r in all_rows
                if (reg_sel == "ALL" or str(r.get("region", "")) == reg_sel)
                and (rte_sel == "ALL" or str(r.get("route", "")) == rte_sel)
            ]
            self._last_rows = rows

            # F-0619/F-0620: sync the economy simulator under its lock, summing
            # counts across groups (max() undercounted multi-group resources).
            # Counts are per-node events; multiply by items_per_gather so the
            # simulator counts actual items, matching telemetry (F-1753).
            # Lifetime rows already count items, so no multiplier there.
            if self.economy_calculator is not None and hasattr(self.economy_calculator, "gathered_counts"):
                try:
                    totals: Dict[str, int] = {}
                    for row in all_rows:
                        res_type = str(row.get("resource_type", "")).strip()
                        cnt = int(row.get("count", 0) or 0)
                        if res_type and cnt > 0:
                            totals[res_type] = totals.get(res_type, 0) + cnt
                    per_event = 1 if range_mode == "all" else int(
                        getattr(self.telemetry, "items_per_gather", 3) or 3
                    )

                    def _apply_sync() -> None:
                        gc = self.economy_calculator.gathered_counts
                        gc.clear()
                        for res_type, cnt in totals.items():
                            gc[res_type] = cnt * per_event

                    sync_lock = getattr(self.economy_calculator, "_lock", None)
                    if sync_lock is not None:
                        with sync_lock:
                            _apply_sync()
                    else:
                        _apply_sync()
                except Exception as e:
                    logger.debug(f"Ekonomi senkronizasyon uyarısı: {e}")

            if range_mode == "all":
                net_kinah = int(metrics.get("lifetime_gold", 0) or 0)
                fmt_net = format_kinah_tr(net_kinah)
                kinah_hr = 0
                fmt_hr = "—"
            else:
                econ_summary = {}
                if self.economy_calculator is not None and hasattr(self.economy_calculator, "get_session_economy_summary"):
                    try:
                        # P1-6 (K-06): oran, toplamın ait olduğu süreye
                        # bölünür — tüm aralık süresine değil. Dilim
                        # seçiliyse dilim süresi; yoksa ekonomi özeti
                        # oturum kapsamlı olduğu için oturum süresi.
                        # Eskiden "bugün/7 gün" aralığında saatlik oran,
                        # gece yarısından beri geçen tüm süreye bölünüp
                        # olduğundan küçük çıkıyordu.
                        slice_start = float(metrics.get("slice_start_ts") or 0.0)
                        slice_end = float(metrics.get("slice_end_ts") or 0.0)
                        if (
                            self._selected_slice is not None
                            and slice_start > 0.0
                            and slice_end > slice_start
                        ):
                            elapsed = max(1.0, slice_end - slice_start)
                        elif duration_sec > 0:
                            elapsed = max(1.0, duration_sec)
                        else:
                            range_start = float(metrics.get("range_start_ts") or 0.0)
                            elapsed = max(1.0, time.time() - range_start) if range_start else max(1.0, duration_sec)
                        effective_hrs = max(1.0 / 3600.0, elapsed / 3600.0)
                        econ_summary = self.economy_calculator.get_session_economy_summary(elapsed_hours=effective_hrs)
                    except Exception as e:
                        logger.error(f"Ekonomi özeti alınırken hata: {e}")

                net_kinah = int(econ_summary.get("net_kinah_earned", 0) or 0)
                fmt_net = str(econ_summary.get("formatted_net_kinah", format_kinah_tr(net_kinah)))
                kinah_hr = int(econ_summary.get("kinah_per_hour", 0) or 0)
                fmt_hr = str(econ_summary.get("formatted_kinah_per_hour", format_kinah_tr(kinah_hr)))

            self.session_info_lbl.setText(
                f"Oturum: {duration_str} | Savaşlar: {combats} | Hatalar: {failures}"
            )

            range_names = {
                "session": "OTURUM",
                "today": "BUGÜN",
                "7d": "SON 7 GÜN",
                "all": "KALICI",
            }
            r_title = range_names.get(self._active_range, "BUGÜN")
            self.range_sec_title.setText(f"{r_title} İSTATİSTİKLERİ (SEÇİLİ ARALIK)")
            self.chart_title_lbl.setText(
                f"VERİM VE SAATLİK TOPLAMA HIZI ({r_title} · 12 DİLİM) — DİLİME TIKLAYIN"
            )
            self.rec_count_lbl.setText(f'<b style="color:#eef1f5;">{gathered + failures}</b> kayıt · {r_title}')

            target_rate = max(1, int(self.target_rate_spin.value()))
            target_pct = min(100, int(round((rate_hr / float(target_rate)) * 100.0)))

            self.card_gathered.update_card(f"{gathered}", f"Ort. Eşleşme Güveni: %{avg_match:.1f}")
            self.card_rate.update_card(
                f"{rate_hr:.1f}",
                f"Hedef {target_rate}/sa · %{target_pct} · aktif saat başına",
                progress_pct=target_pct,
            )
            # F-0708: no attempts yet -> show "—", never a fabricated 100%.
            if success_pct is None:
                self.card_success.update_card("%—", "henüz deneme yok")
            else:
                self.card_success.update_card(f"%{success_pct:.1f}", f"{failures} başarısız deneme")
            self.card_walk.update_card(f"{avg_walk:.2f}sn", f"Temizlenen savaş: {combats}")

            self.card_kinah_total.update_card(f"{fmt_net}", "Seçili aralıkta kazanılan")
            self.card_kinah_rate.update_card(f"{fmt_hr}/sa", "Aktif saat başına kinah")

            buckets = metrics.get("hourly_buckets", [])
            if gathered + failures == 0:
                # Kayıt yoksa grafik sahte sıfır çubukları değil boş durumu çizer.
                self.chart_widget.set_buckets([])
            else:
                self.chart_widget.set_buckets(buckets)

            self._update_insights(rows, avg_match)
            self._populate_table(rows)
            self._update_detail_inspector()
        except Exception as exc:
            logger.error(f"refresh_metrics sırasında hata: {exc}", exc_info=True)
            # F-0709: never leave the UI silently showing stale data.
            try:
                self.session_info_lbl.setText("⚠ Yenileme hatası — bayat veri gösteriliyor")
            except Exception:
                pass

    def _update_insights(self, rows: List[Dict[str, Any]], avg_match: float) -> None:
        if not rows:
            self.ins_rota.update_insight("—", "Henüz veri yok")
            self.ins_kinah.update_insight("—", "Henüz veri yok")
            self.ins_dikkat.update_insight("—", "Sorun tespit edilmedi")
            return
        # EN VERİMLİ ROTA (prototip): rota başına ortalama kinah; bilinen
        # broker fiyatı yoksa adede göre en çok toplayan rota gösterilir.
        route_stats: Dict[str, List[int]] = {}
        for r in rows:
            kv = self._row_kinah(r)
            if kv is None:
                continue
            rte = str(r.get("route", "—") or "—")
            acc = route_stats.setdefault(rte, [0, 0])
            acc[0] += int(kv)
            acc[1] += int(r.get("count", 0) or 0)
        if route_stats:
            best_route, (tot_k, tot_c) = max(
                route_stats.items(), key=lambda kv: kv[1][0] / max(1, kv[1][1])
            )
            self.ins_rota.update_insight(
                str(best_route),
                f"toplama başına ort. {tot_k // max(1, tot_c)} kinah · {tot_c} toplama",
            )
        else:
            by_count = sorted(rows, key=lambda r: int(r.get("count", 0) or 0), reverse=True)
            best_r = by_count[0]
            self.ins_rota.update_insight(
                f"{best_r.get('region', '—')} · {best_r.get('route', 'Varsayılan')}",
                f"{int(best_r.get('count', 0) or 0)} adet · fiyat bilgisi yok, adede göre",
            )
        # F-0546: 'most Kinah' is ranked by known-price Kinah value, not by
        # item count; with no known price for any row, say so.
        priced = [(self._row_kinah(r), r) for r in rows]
        priced = [(v, r) for v, r in priced if v is not None and v > 0]
        if priced:
            best_kinah, best_kinah_row = max(priced, key=lambda kv: kv[0])
            self.ins_kinah.update_insight(
                str(best_kinah_row.get("resource_type", "Materyal")),
                f"{format_kinah_tr(best_kinah)} Kinah · {int(best_kinah_row.get('count', 0) or 0)} adet",
            )
        else:
            self.ins_kinah.update_insight("—", "Fiyat bilgisi yok")
        # DİKKAT (prototip): en yavaş grup ve en düşük güvenli grup.
        # Az kayıtlı gruplar (3'ten az) gürültü olmasın diye önce onlara bakılır.
        cands = [r for r in rows if int(r.get("count", 0) or 0) >= 3] or list(rows)
        slowest = max(cands, key=lambda r: float(r.get("avg_walk_sec", 0.0) or 0.0))
        lowest = min(cands, key=lambda r: float(r.get("avg_match_pct", 0.0) or 0.0))
        slow_walk = float(slowest.get("avg_walk_sec", 0.0) or 0.0)
        low_conf = float(lowest.get("avg_match_pct", 0.0) or 0.0)
        if slowest is lowest:
            self.ins_dikkat.update_insight(
                f"{slowest.get('resource_type', '—')} ({slowest.get('route', '—')})",
                f"yürüme {slow_walk:.2f}sn · güven %{low_conf:.0f} — en yavaş ve en düşük",
            )
        else:
            self.ins_dikkat.update_insight(
                f"{slowest.get('resource_type', '—')} ({slowest.get('route', '—')})",
                f"yürüme {slow_walk:.2f}sn en yavaş · "
                f"{lowest.get('resource_type', '—')} güven %{low_conf:.0f} en düşük",
            )

    def _lifetime_kills(self) -> Optional[int]:
        try:
            return int(get_lifetime_stats().snapshot().get("monsters_killed", 0) or 0)
        except Exception:
            return None

    def _update_detail_inspector(self) -> None:
        metrics = self._last_metrics
        rows = self._last_rows
        kills = self._lifetime_kills()
        kills_txt = f"{kills}" if kills is not None else "—"

        # -- Tek olay seçimi (prototip: "Olay Kaydı" detayı) -------------
        if getattr(self, "_selected_event", None):
            kind, ev = self._selected_event
            try:
                ts = float(ev.get("timestamp", 0.0) or 0.0)
                ts_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "—"
            except (TypeError, ValueError, OSError):
                ts_str = "—"
            if kind == "gather":
                self.detail_kicker_lbl.setText("SEÇİLİ OLAY")
                self.detail_title_lbl.setText(str(ev.get("resource_type", "Toplama")))
                self.detail_sub_lbl.setText(
                    f"{ts_str} · {ev.get('region', '—')} · {ev.get('route', '—')}"
                )
                pairs = [
                    ("Kaynak", str(ev.get("resource_type", "—"))),
                    ("Rota", str(ev.get("route", "—"))),
                    ("Sonuç", "Toplandı"),
                    ("Eşleşme güveni", f"%{int(ev.get('match_pct', 0) or 0)}"),
                    ("Yürüme süresi", f"{float(ev.get('walk_duration_sec', 0.0) or 0.0):.2f} sn"),
                    ("Kazanılan kinah", self._event_kinah_text(ev)),
                ]
            elif kind == "fail":
                self.detail_kicker_lbl.setText("SEÇİLİ OLAY")
                self.detail_title_lbl.setText("Başarısız Deneme")
                self.detail_sub_lbl.setText(f"{ts_str} · Node #{ev.get('node_id', '—')}")
                pairs = [
                    ("Tür", "Başarısız deneme"),
                    ("Node", f"#{ev.get('node_id', '—')}"),
                    ("Sebep", str(ev.get("reason", "timeout"))),
                    ("Zaman", ts_str),
                    ("Öldürülen canavar (kalıcı)", kills_txt),
                    ("Not", "Başarısız deneme kaynak taşımaz"),
                ]
            else:  # combat
                self.detail_kicker_lbl.setText("SEÇİLİ OLAY")
                self.detail_title_lbl.setText("Savaş Olayı")
                self.detail_sub_lbl.setText(f"{ts_str} · Sonuç: {ev.get('outcome', '—')}")
                pairs = [
                    ("Tür", "Savaş"),
                    ("Sonuç", str(ev.get("outcome", "—"))),
                    ("Süre", f"{float(ev.get('duration_sec', 0.0) or 0.0):.2f} sn"),
                    ("Zaman", ts_str),
                    ("Öldürülen canavar (kalıcı)", kills_txt),
                    ("Not", "Savaş olayı rota/kaynak taşımaz"),
                ]
            for (kl, vl), (k_txt, v_txt) in zip(self.detail_kv_labels, pairs):
                kl.setText(k_txt)
                vl.setText(v_txt)
            self.btn_group_csv.setVisible(False)
            self.btn_clear_selection.setText("← Özete Dön")
            self.btn_clear_selection.setVisible(True)
            self._populate_detail_events()
            return

        if self._selected_group_key:
            matched = None
            for r in rows:
                k = f"{r.get('resource_type', '')}|{r.get('region', '')}|{r.get('route', '')}"
                if k == self._selected_group_key:
                    matched = r
                    break
            if matched is not None:
                self.detail_kicker_lbl.setText("SEÇİLİ KAYNAK GRUBU")
                self.detail_title_lbl.setText(str(matched.get("resource_type", "Kaynak")))
                self.detail_sub_lbl.setText(f"{matched.get('region', '—')} · {matched.get('route', '—')}")
                cnt = int(matched.get("count", 0))
                share = float(matched.get("share_pct", 0.0))
                walk = float(matched.get("avg_walk_sec", 0.0))
                pairs = [
                    ("Toplanan adet", f"{cnt} (%{share:.1f})"),
                    ("Bölge", str(matched.get("region", "—"))),
                    ("Rota", str(matched.get("route", "—"))),
                    ("Ort. yürüme süresi", f"{walk:.2f} sn"),
                    ("Ort. eşleşme güveni", f"%{float(metrics.get('avg_match_pct', 0.0) or 0.0):.1f}"),
                    # F-0710: no hard-coded "Durum: Aktif"; show lifetime kills.
                    ("Öldürülen canavar (kalıcı)", kills_txt),
                ]
                for (kl, vl), (k_txt, v_txt) in zip(self.detail_kv_labels, pairs):
                    kl.setText(k_txt)
                    vl.setText(v_txt)
                self.btn_group_csv.setVisible(True)
                self.btn_clear_selection.setText("Seçimi Temizle")
                self.btn_clear_selection.setVisible(True)
                self._populate_detail_events()
                return

        self.btn_group_csv.setVisible(False)
        self.btn_clear_selection.setVisible(False)
        range_titles = {
            "session": "Oturum Özeti",
            "today": "Bugün Özeti",
            "7d": "Son 7 Gün Özeti",
            "all": "Kalıcı Özet",
        }
        self.detail_kicker_lbl.setText("ARALIK ÖZETİ")
        self.detail_title_lbl.setText(range_titles.get(self._active_range, "Bugün Özeti"))
        gathered = int(metrics.get("gathered_total", 0))
        failures = int(metrics.get("failed_total", 0))
        combats = int(metrics.get("combat_count", 0))
        res_Count = len({str(r.get("resource_type", "")) for r in rows})
        rte_Count = len({str(r.get("route", "")) for r in rows})
        self.detail_sub_lbl.setText(f"{gathered + failures} kayıt · seçili filtre")
        # v20-istatistik (R1): özet satırı artık aralığın kill'ini gösterir.
        # Eskiden her aralıkta kalıcı kill (89) basılıyordu — "Bugün Özeti"
        # kalıcı sayıyı bugünün sayısı gibi gösteriyordu. Kalıcı kill
        # yalnızca "Kalıcı" aralığında, "(kalıcı)" etiketiyle görünür.
        kills_range = metrics.get("kills_in_range")
        kills_range_txt = f"{int(kills_range)}" if kills_range is not None else "—"
        kills_range_label = str(metrics.get("kills_label") or "Öldürülen canavar")
        pairs = [
            ("Toplam deneme", f"{gathered + failures} ({gathered} başarılı)"),
            ("Farklı kaynak türü", f"{res_Count}"),
            ("Farklı rota", f"{rte_Count}"),
            ("Savaş / temizlenen", f"{combats}"),
            # F-0710: the 5th label is "Öldürülen canavar" — show kills, not
            # a second copy of the gather count.
            (kills_range_label, kills_range_txt),
            ("Başarısız deneme", f"{failures}"),
        ]
        for (kl, vl), (k_txt, v_txt) in zip(self.detail_kv_labels, pairs):
            kl.setText(k_txt)
            vl.setText(v_txt)
        self._populate_detail_events()

    def _populate_detail_events(self) -> None:
        """Detay panelindeki SON OLAYLAR listesini doldurur (en fazla 6).

        Kalıcı aralıkta olay kaydı tutulmadığı için liste boş gösterilir.
        """
        try:
            lst = self.detail_events_list
            lst.blockSignals(True)
            lst.clear()
            mode = "all" if str(getattr(self, "_active_range", "today")) == "all" else ""
            events = [] if mode == "all" else self._recent_events(6)
            if not events:
                empty = QListWidgetItem(
                    "henüz kayıt yok" if mode != "all"
                    else "Kalıcı aralıkta olay kaydı tutulmaz"
                )
                empty.setFlags(Qt.ItemFlag.NoItemFlags)
                lst.addItem(empty)
            else:
                for kind, ev in events:
                    try:
                        ts = float(ev.get("timestamp", 0.0) or 0.0)
                        ts_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "--:--:--"
                    except (TypeError, ValueError, OSError):
                        ts_str = "--:--:--"
                    if kind == "gather":
                        text = (
                            f"{ts_str} · {ev.get('resource_type', '—')} · "
                            f"{ev.get('route', '—')} — Toplandı"
                        )
                    elif kind == "fail":
                        text = f"{ts_str} · Başarısız — {ev.get('reason', 'timeout')}"
                    else:
                        text = (
                            f"{ts_str} · Savaş {ev.get('outcome', '—')} — "
                            f"{float(ev.get('duration_sec', 0.0) or 0.0):.1f} sn"
                        )
                    item = QListWidgetItem(text)
                    item.setData(Qt.ItemDataRole.UserRole, (kind, dict(ev)))
                    lst.addItem(item)
            lst.blockSignals(False)
        except Exception as exc:
            logger.debug(f"_populate_detail_events hatası: {exc}")

    def _populate_table(self, rows: List[Dict[str, Any]]) -> None:
        try:
            # F-0618: enrich rows with the values the KİNAH / ORT. GÜVEN
            # columns sort by — the raw rows carry neither key, so sorting
            # those columns was a no-op (every key defaulted to 0).
            enriched: List[Dict[str, Any]] = []
            for r in rows:
                if not isinstance(r, dict):
                    continue
                item = dict(r)
                kinah_val = self._row_kinah(item)
                item["kinah_value"] = int(kinah_val) if kinah_val is not None else 0
                try:
                    item["avg_conf"] = float(item.get("avg_match_pct", 0.0) or 0.0)
                except (TypeError, ValueError):
                    item["avg_conf"] = 0.0
                enriched.append(item)
            sorted_rows = sorted(
                enriched,
                key=lambda r: (
                    str(r.get(self._sort_col, ""))
                    if self._sort_col in ("resource_type", "region", "route")
                    else float(r.get(self._sort_col, 0.0) or 0.0)
                ),
                reverse=(self._sort_dir < 0),
            )
            self.table.blockSignals(True)
            self.table.clearSpans()
            self.table.setRowCount(len(sorted_rows))
            if not sorted_rows:
                # Boş durum (prototip): sahte satır yok, tek bilgilendirme satırı.
                self.table.setRowCount(1)
                empty_it = QTableWidgetItem(
                    "henüz kayıt yok — aralığı genişletin veya bölge/rota filtresini sıfırlayın"
                )
                empty_it.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
                empty_it.setForeground(QBrush(QColor("#6b7280")))
                self.table.setItem(0, 0, empty_it)
                self.table.setSpan(0, 0, 1, 9)
                self.table.blockSignals(False)
                return
            avg_match = float(self._last_metrics.get("avg_match_pct", 0.0) or 0.0)

            for r_idx, row_data in enumerate(sorted_rows):
                self.table.setRowHeight(r_idx, 36)
                res_name = str(row_data.get("resource_type", "Bilinmiyor"))
                reg_name = str(row_data.get("region", "Bilinmiyor"))
                rte_name = str(row_data.get("route", "Varsayılan"))
                cnt = int(row_data.get("count", 0))
                share_pct = float(row_data.get("share_pct", 0.0))
                walk_sec = float(row_data.get("avg_walk_sec", 0.0))

                # F-0547: only a known broker price yields a Kinah figure;
                # an unknown material shows "fiyat yok", never a fabricated
                # 10.000 Kinah/unit default.
                unit_price = self._known_unit_price(res_name)
                if unit_price is not None and unit_price > 0:
                    est_kinah = format_kinah_tr(cnt * unit_price)
                elif cnt > 0:
                    est_kinah = "fiyat yok"
                else:
                    est_kinah = "0 Kinah"

                # Real match confidence per resource or fallback to overall average
                res_match = float(row_data.get("avg_match_pct", 0.0) or avg_match)
                conf_str = f"%{res_match:.1f}" if res_match > 0.0 else "%0.0"

                # Real gather event timestamp lookup
                last_ts_str = str(row_data.get("last_gathered_str") or row_data.get("time_str") or "").strip()
                if not last_ts_str and hasattr(self, "telemetry") and self.telemetry is not None:
                    events = getattr(self.telemetry, "gather_events", None) or []
                    for ev in reversed(events):
                        if (
                            str(ev.get("resource_type", "")).strip() == res_name
                            and str(ev.get("region", "")).strip() == reg_name
                        ):
                            ts = ev.get("timestamp")
                            if ts:
                                last_ts_str = datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
                                break
                if not last_ts_str:
                    last_ts_str = "—"

                group_key = f"{res_name}|{reg_name}|{rte_name}"

                it_res = QTableWidgetItem(res_name)
                it_res.setData(Qt.ItemDataRole.UserRole, group_key)
                self.table.setItem(r_idx, 0, it_res)

                self.table.setItem(r_idx, 1, QTableWidgetItem(reg_name))
                self.table.setItem(r_idx, 2, QTableWidgetItem(rte_name))

                it_cnt = QTableWidgetItem(str(cnt))
                it_cnt.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
                self.table.setItem(r_idx, 3, it_cnt)

                # F-0711: the PAY % cell is a widget only — a setItem() text
                # item underneath it was invisible (widget covers the cell).
                self.table.setCellWidget(r_idx, 4, _ShareBarCellWidget(share_pct, self.table))

                it_walk = QTableWidgetItem(f"{walk_sec:.2f}sn")
                it_walk.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
                self.table.setItem(r_idx, 5, it_walk)

                it_kinah = QTableWidgetItem(est_kinah)
                it_kinah.setForeground(QBrush(QColor("#f5b50a")))
                it_kinah.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
                self.table.setItem(r_idx, 6, it_kinah)

                it_conf = QTableWidgetItem(conf_str)
                it_conf.setForeground(QBrush(QColor("#34d399")))
                it_conf.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
                self.table.setItem(r_idx, 7, it_conf)

                it_last = QTableWidgetItem(last_ts_str)
                it_last.setForeground(QBrush(QColor("#9aa3af")))
                it_last.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
                self.table.setItem(r_idx, 8, it_last)
            self.table.blockSignals(False)
        except Exception as exc:
            self.table.blockSignals(False)
            logger.error(f"_populate_table sırasında hata: {exc}", exc_info=True)

    def on_export_csv(self) -> None:
        """Kullanıcının seçtiği veya varsayılan CSV dosyasına istatistikleri aktarır."""
        try:
            default_path = os.path.abspath("logs/session_stats.csv")
            os.makedirs(os.path.dirname(default_path), exist_ok=True)
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "CSV Dışa Aktar",
                default_path,
                "CSV Dosyaları (*.csv);;Tüm Dosyalar (*)",
            )
            if not file_path:
                return

            ok = self.telemetry.save_session_report(json_path="", csv_path=file_path)
            if ok:
                QMessageBox.information(self, "CSV Dışa Aktar", f"CSV raporu başarıyla dışa aktarıldı:\n{file_path}")
            else:
                QMessageBox.warning(self, "CSV Dışa Aktar", "CSV raporu dışa aktarılamadı.")
        except Exception as exc:
            logger.error(f"on_export_csv hatası: {exc}", exc_info=True)

    def on_save_json(self) -> None:
        """Tam oturum telemetri raporunu JSON olarak kaydeder."""
        try:
            default_path = os.path.abspath("logs/telemetry_history.json")
            os.makedirs(os.path.dirname(default_path), exist_ok=True)
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "JSON Telemetri Raporunu Kaydet",
                default_path,
                "JSON Dosyaları (*.json);;Tüm Dosyalar (*)",
            )
            if not file_path:
                return

            ok = self.telemetry.save_session_report(json_path=file_path, csv_path="")
            if ok:
                QMessageBox.information(self, "JSON Rapor Kaydet", f"JSON raporu başarıyla kaydedildi:\n{file_path}")
            else:
                QMessageBox.warning(self, "JSON Rapor Kaydet", "JSON raporu kaydedilemedi.")
        except Exception as exc:
            logger.error(f"on_save_json hatası: {exc}", exc_info=True)

    def on_export_group_csv(self) -> None:
        """Seçili kaynak grubunun olaylarını CSV olarak kaydeder (prototip: 'Grubu CSV İndir')."""
        try:
            if not self._selected_group_key:
                return
            parts = str(self._selected_group_key).split("|")
            if len(parts) != 3:
                return
            res_name, reg_name, rte_name = parts
            events = [kind_ev for kind_ev in self._recent_events(100000) if kind_ev[0] == "gather"]
            # _recent_events grup filtresi zaten uygular; yine de doğrula.
            events = [
                ev for _k, ev in events
                if str(ev.get("resource_type", "")) == res_name
                and str(ev.get("region", "")) == reg_name
                and str(ev.get("route", "")) == rte_name
            ]
            safe = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in res_name).strip("-") or "grup"
            default_path = os.path.abspath(os.path.join("logs", f"gatherion-{safe}.csv"))
            os.makedirs(os.path.dirname(default_path), exist_ok=True)
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Grubu CSV İndir", default_path, "CSV Dosyaları (*.csv);;Tüm Dosyalar (*)"
            )
            if not file_path:
                return
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f, delimiter=";")
                w.writerow(["Kaynak", "Bolge", "Rota", "Adet", "Ort. Guven", "Ort. Yurume (sn)", "Tahmini Kinah"])
                total_kinah = 0
                for ev in events:
                    price = self._known_unit_price(res_name)
                    items = int(ev.get("items_gathered", 1) or 1)
                    kinah = (price * items) if price is not None else 0
                    total_kinah += kinah
                    w.writerow([
                        res_name, reg_name, rte_name, items,
                        int(ev.get("match_pct", 0) or 0),
                        f"{float(ev.get('walk_duration_sec', 0.0) or 0.0):.2f}",
                        kinah if price is not None else "fiyat yok",
                    ])
                w.writerow([])
                w.writerow(["Saat", "Sonuc", "Guven", "Yurume (sn)", "Kinah", "Canavar"])
                for ev in events:
                    price = self._known_unit_price(res_name)
                    items = int(ev.get("items_gathered", 1) or 1)
                    kinah = (price * items) if price is not None else 0
                    ts = ev.get("iso_time") or datetime.fromtimestamp(
                        float(ev.get("timestamp", 0.0) or 0.0)
                    ).isoformat(timespec="seconds")
                    w.writerow([ts, "ok", int(ev.get("match_pct", 0) or 0),
                                f"{float(ev.get('walk_duration_sec', 0.0) or 0.0):.2f}",
                                kinah if price is not None else "fiyat yok", "—"])
            self._append_log_record(
                "INFO", f"Grup CSV indirildi — {res_name} ({len(events)} olay, tahmini {format_kinah_tr(total_kinah)})"
            )
            QMessageBox.information(self, "Grubu CSV İndir", f"Grup CSV kaydedildi:\n{file_path}")
        except Exception as exc:
            logger.error(f"on_export_group_csv hatası: {exc}", exc_info=True)

    def on_reset_stats(self) -> None:
        """Yalnızca oturum istatistiklerini sıfırlar; kalıcı lifetime dokunulmaz."""
        try:
            reply = QMessageBox.question(
                self,
                "Oturumu Sıfırla",
                "Bu oturumun geçici sayaçları sıfırlansın mı?\n\n"
                "Kalıcı (ömür boyu) materyal / altın / canavar toplamları silinmez.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self.telemetry.reset_session()
            if self.economy_calculator is not None and hasattr(self.economy_calculator, "reset_session"):
                self.economy_calculator.reset_session()
            self.refresh_metrics()
            logger.info("Kullanıcı tarafından oturum istatistikleri sıfırlandı (lifetime korundu).")
        except Exception as exc:
            logger.error(f"on_reset_stats hatası: {exc}", exc_info=True)

    def on_reset_lifetime_stats(self) -> None:
        """Kalıcı lifetime istatistiklerini yalnızca açık onayla sıfırlar."""
        try:
            reply = QMessageBox.warning(
                self,
                "Kalıcı İstatistikleri Sıfırla",
                "Ömür boyu materyal, altın ve canavar sayaçları KALICI olarak silinecek.\n\n"
                "Bu işlem geri alınamaz. Devam edilsin mi?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            get_lifetime_stats().reset(confirm=True)
            self.refresh_metrics()
            QMessageBox.information(self, "Kalıcı İstatistikler", "Kalıcı istatistikler sıfırlandı.")
            logger.info("Kullanıcı onayıyla lifetime istatistikleri sıfırlandı.")
        except Exception as exc:
            logger.error(f"on_reset_lifetime_stats hatası: {exc}", exc_info=True)

    def closeEvent(self, event: Any) -> None:
        try:
            if self._refresh_timer.isActive():
                self._refresh_timer.stop()
        except Exception as exc:
            logger.error(f"closeEvent hatası: {exc}")
        try:
            if self._log_handler is not None:
                for name in _LIVE_LOG_SOURCES:
                    try:
                        lg = logging.getLogger(name)
                        if self._log_handler in lg.handlers:
                            lg.removeHandler(self._log_handler)
                    except Exception:
                        pass
                self._log_handler = None
        except Exception:
            pass
        super().closeEvent(event)

    def get_combat_and_consumable_analytics(self) -> Dict[str, Any]:
        """Returns session combat, potion usage, buff scroll refresh, and selective gather metrics.
        F-0548: keys the telemetry does not track are None (unknown), never
        a fabricated 0 — and selective_gather_hits is never silently equated
        with gathered_total. success_rate_pct is None when unknown.
        """
        metrics = dict(self.__dict__.get("_last_metrics") or {})
        telemetry = self.__dict__.get("telemetry")
        if not metrics and telemetry is not None and hasattr(telemetry, "get_live_metrics"):
            try:
                metrics = dict(telemetry.get_live_metrics() or {})
            except Exception:
                metrics = {}

        def _opt_int(key: str) -> Optional[int]:
            if key not in metrics or metrics.get(key) is None:
                return None
            try:
                return int(metrics.get(key))
            except (TypeError, ValueError):
                return None

        def _opt_float(key: str) -> Optional[float]:
            if key not in metrics or metrics.get(key) is None:
                return None
            try:
                return float(metrics.get(key))
            except (TypeError, ValueError):
                return None

        return {
            "combat_count": int(metrics.get("combat_count", 0) or 0),
            "gathered_total": int(metrics.get("gathered_total", 0) or 0),
            "failed_total": int(metrics.get("failed_total", 0) or 0),
            "hp_potions_used": _opt_int("hp_potions_used"),
            "mp_potions_used": _opt_int("mp_potions_used"),
            "buff_scrolls_used": _opt_int("buff_scrolls_used"),
            "selective_gather_hits": _opt_int("selective_gather_hits"),
            "success_rate_pct": _opt_float("success_rate_pct"),
        }

    def export_analytics_snapshot_dict(self) -> Dict[str, Any]:
        """Exports a unified dictionary combining session KPIs, lifetime stats, and consumable analytics."""
        snap_lt = {}
        try:
            snap_lt = get_lifetime_stats().snapshot()
        except Exception:
            pass
        return {
            "active_range": str(self.__dict__.get("_active_range", "today")),
            "combat_and_consumables": self.get_combat_and_consumable_analytics(),
            "lifetime": snap_lt,
            "rows_count": len(self.__dict__.get("_last_rows") or []),
            "ready": bool(self.__dict__.get("_last_metrics") or snap_lt),
        }

    def get_analytics_dialog_health_summary(self) -> dict:
        """F-0621: real check — the attached telemetry must return a live
        metrics dict with the core keys. No telemetry / broken metrics ->
        ready False with the reason, never a constant True."""
        try:
            tel = self.__dict__.get("telemetry")
            if tel is None or not hasattr(tel, "get_live_metrics"):
                return {"analytics_dialog_ready": False, "error": "telemetry yok"}
            metrics = tel.get_live_metrics()
            ok = (
                isinstance(metrics, dict)
                and "gathered_total" in metrics
                and "success_rate_pct" in metrics
            )
            return {
                "analytics_dialog_ready": bool(ok),
                "telemetry_live_metrics_ok": bool(ok),
                "gathered_total": metrics.get("gathered_total") if isinstance(metrics, dict) else None,
            }
        except Exception as exc:
            return {"analytics_dialog_ready": False, "error": str(exc)}

    def get_analytics_dialog_ready_flag(self) -> dict:
        ready = bool(self.get_analytics_dialog_health_summary().get("analytics_dialog_ready"))
        return {"module": "analytics_dialog", "ready": ready}

    def get_analytics_dialog_version_info(self) -> dict:
        ready = bool(self.get_analytics_dialog_health_summary().get("analytics_dialog_ready"))
        return {"module": "analytics_dialog", "version": __version__, "ready": ready}

    def get_analytics_dialog_diagnostic_snapshot(self) -> dict:
        health = self.get_analytics_dialog_health_summary()
        ready = bool(health.get("analytics_dialog_ready"))
        return {
            "module": "analytics_dialog",
            "status": "ok" if ready else "degraded",
            "ready": ready,
            "health": health,
        }


def get_analytics_dialog_capabilities() -> list:
    return ["analytics_dialog", "telemetry_visualization", "diagnostic_snapshot"]


def validate_analytics_dialog_contract() -> bool:
    """F-0621: real check — a fresh TelemetryEngine must produce a live
    metrics dict with the core keys."""
    try:
        engine = TelemetryEngine()
        metrics = engine.get_live_metrics()
        return (
            isinstance(metrics, dict)
            and "gathered_total" in metrics
            and "success_rate_pct" in metrics
        )
    except Exception:
        return False


def get_analytics_dialog_baseline_metrics() -> dict:
    return {"module": "analytics_dialog", "contract_valid": validate_analytics_dialog_contract(), "capability_count": len(get_analytics_dialog_capabilities())}


def get_analytics_dialog_readiness_status() -> dict:
    return {"module": "analytics_dialog", "ready": validate_analytics_dialog_contract(), "metrics": get_analytics_dialog_baseline_metrics()}


def get_analytics_dialog_audit_record() -> dict:
    return {"module": "analytics_dialog", "audit_passed": validate_analytics_dialog_contract(), "readiness": get_analytics_dialog_readiness_status()}


def get_analytics_dialog_operational_digest() -> dict:
    return {"module": "analytics_dialog", "operational": validate_analytics_dialog_contract(), "audit": get_analytics_dialog_audit_record()}


def get_analytics_dialog_telemetry_payload() -> dict:
    return {"module": "analytics_dialog", "telemetry_ready": validate_analytics_dialog_contract(), "digest": get_analytics_dialog_operational_digest()}


def get_analytics_dialog_integrity_seal() -> dict:
    return {"module": "analytics_dialog", "sealed": validate_analytics_dialog_contract(), "telemetry": get_analytics_dialog_telemetry_payload()}


def get_analytics_dialog_compliance_certificate() -> dict:
    return {"module": "analytics_dialog", "certified": validate_analytics_dialog_contract(), "seal": get_analytics_dialog_integrity_seal()}


def get_analytics_dialog_readiness_probe() -> dict:
    return {"module": "analytics_dialog", "ready": validate_analytics_dialog_contract(), "certificate": get_analytics_dialog_compliance_certificate()}


def get_analytics_dialog_audit_summary() -> dict:
    return {"module": "analytics_dialog", "audit_passed": validate_analytics_dialog_contract(), "probe": get_analytics_dialog_readiness_probe()}
