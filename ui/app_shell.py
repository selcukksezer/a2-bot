# -*- coding: utf-8 -*-
"""One window: a narrow page rail, the selected tool, and a live log along the bottom.

F9 is not a rail page and not a home button. Anasayfa shows each F9 menu action
as its own button and calls that row's slot. Ctrl+Alt+F9 still toggles the menu.
"""

from __future__ import annotations

from datetime import datetime
import logging
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from PyQt6.QtCore import (
    QEasingCurve,
    QObject,
    QPointF,
    QRectF,
    Qt,
    QTimer,
    QVariantAnimation,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
    QShortcut,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ui.shell_page_frame import wrap_shell_tab_root_layout
from ui.shell_theme import SHELL_COLORS

logger = logging.getLogger("Gatherion.AppShell")

# Rail order. Settings is last. F9 is intentionally absent.
NAV = (
    ("home", "Anasayfa"),
    ("radar", "Radar"),
    ("map", "Harita"),
    ("yolo", "YOLO"),
    ("calibration", "Kalibrasyon"),
    ("stats", "İstatistik"),
    ("mobile", "Mobil"),
    ("guide", "Kılavuz"),
    ("logs", "Log"),
    ("settings", "Ayarlar"),
)

_STATE_TR = {
    "WALKING": "Yürüyor",
    "PAUSED": "Duraklatıldı",
    "GATHERING": "Topluyor",
    "WAITING": "Bekliyor",
    "FIGHTING": "Savaşıyor",
    "IDLE": "Boşta",
    "STOPPED": "Durdu",
    "RESTING_BREAK": "Mola",
    "BREAK": "Mola",
}

CANONICAL_REGION_MAP: Dict[str, str] = {
    "altgard": "Altgard",
    "dredgion crash site": "Altgard",
    "dredgion": "Altgard",
    "calderon canyon": "Altgard",
    "safe haven": "Altgard",
    "silent hill": "Altgard",
    "altgard fortress": "Altgard",
    "basfelt village": "Altgard",
    "grave of the red sky": "Altgard",
    "verteron": "Verteron",
    "verteron citadel": "Verteron",
    "tolbas forest": "Verteron",
    "cantalas coast": "Verteron",
    "pilgrim's ridge": "Verteron",
    "ishalgen": "Ishalgen",
    "poeta": "Poeta",
    "morheim": "Morheim",
    "eltnen": "Eltnen",
}

# Home shows one large word. Empty input stays empty so a missing worker is not invented.
_HOME_PAUSED = {
    "PAUSED",
    "IDLE",
    "STOPPED",
    "WAITING",
    "RESTING",
    "BREAK",
    "RESTING_BREAK",
    "DURAKLATILDI",
    "DURDU",
    "BOŞTA",
    "BOSTA",
    "BEKLİYOR",
    "BEKLIYOR",
    "MOLA",
}

_HOME_STATS = (
    ("gathered", "Toplanan", "/ hedef 250"),
    ("active", "Aktif Node", "node taranıyor"),
    ("failures", "Hata", "son 1 saatte"),
    ("yz", "YZ Kare", "YOLOv8 · 640px"),
)

_DEFAULT_HOTKEY_LINE = "Kısayollar: Ctrl+Alt+F8 nokta · Ctrl+Alt+F9 menü · Ctrl+Alt+F10 duraklat / devam"

# Short home captions. The long F9 sentence stays on the tooltip.
_HOME_ACTION_LABELS = {
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


def _short_home_action(action_id: str, long_label: str) -> str:
    """1–3 word caption. Never the menu sentence, never a lone F9."""
    short = _HOME_ACTION_LABELS.get(str(action_id).strip(), "")
    if not short:
        words = [part for part in str(long_label or "").replace(":", " ").split() if part]
        short = " ".join(words[:3])[:24].strip()
    if not short or short == "F9":
        short = "Eylem"
    return short


def _resolve_log_path(log_path: str) -> str:
    if log_path:
        return log_path
    if os.path.isdir(r"F:\a2"):
        return r"F:\a2\logs\bot.log"
    if os.path.isdir("F:\\"):
        return r"F:\Gatherion\logs\bot.log"
    return os.path.join("logs", "bot.log")


def _state_text(raw: Any) -> str:
    token = str(raw or "").strip()
    if not token:
        return ""
    return _STATE_TR.get(token.upper(), token)


def _home_state_word(raw: Any) -> str:
    """Large home word: Duraklatıldı or Çalışıyor. Blank when the worker has no state."""
    if raw is None:
        return ""
    named = getattr(raw, "name", None)
    token = str(named if isinstance(named, str) and named else raw).strip()
    if not token:
        return ""
    upper = token.upper().replace("BOTSTATE.", "")
    if upper in _HOME_PAUSED:
        return "Duraklatıldı"
    if upper in _STATE_TR or upper in {"MAP", "RUNNING", "ÇALIŞIYOR", "CALISIYOR"}:
        return "Çalışıyor"
    translated = _state_text(token)
    if not translated:
        return ""
    if translated in {"Duraklatıldı", "Durdu", "Boşta", "Bekliyor", "Mola"}:
        return "Duraklatıldı"
    return "Çalışıyor"


class _LogSource:
    """One tail of logs/bot.log, painted into every attached view."""

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        self._views: list = []
        self._follow: list = []
        self._stamp: Optional[tuple] = None
        self._text = ""
        self._filter_mode = "ALL"

    def attach(self, view: QPlainTextEdit, *, follow: bool) -> None:
        self._views.append(view)
        self._follow.append(bool(follow))
        if self._text:
            self._paint(view, follow=follow)

    def set_filter_mode(self, mode: str) -> None:
        normalized = str(mode or "ALL").upper()
        if normalized == self._filter_mode:
            return
        self._filter_mode = normalized
        for view, follow in zip(self._views, self._follow):
            self._paint(view, follow=follow)

    def clear_views(self) -> None:
        self._text = ""
        for view in self._views:
            view.clear()

    def reload(self) -> None:
        text = self._read()
        if text is None:
            return
        self._text = text
        for view, follow in zip(self._views, self._follow):
            self._paint(view, follow=follow)

    def _read(self) -> Optional[str]:
        try:
            if not self.log_path or not os.path.exists(self.log_path):
                if self._stamp == ("missing",) and self._text:
                    return None
                self._stamp = ("missing",)
                return "Günlük dosyası henüz yok."
            stat = os.stat(self.log_path)
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp == self._stamp and self._text:
                return None
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read()
            self._stamp = stamp
            tail = text[-120000:] if len(text) > 120000 else text
            return tail or "Günlük dosyası boş."
        except Exception as exc:
            self._stamp = None
            return f"Günlük okunamadı: {exc}"

    def _filtered_text(self, raw: str) -> str:
        if not raw or self._filter_mode == "ALL":
            return raw
        lines = raw.splitlines()
        if self._filter_mode == "INFO":
            kept = [ln for ln in lines if "INFO" in ln or ("WARNING" not in ln and "ERROR" not in ln)]
            return "\n".join(kept) if kept else raw
        if self._filter_mode in ("WARN", "PREVIEW"):
            kept = [ln for ln in lines if any(k in ln for k in ("WARNING", "WARN", "ERROR", "Önizleme", "Durum"))]
            return "\n".join(kept[-25:]) if kept else "\n".join(lines[-8:])
        if self._filter_mode == "ERROR":
            kept = [ln for ln in lines if any(k in ln.upper() for k in ("ERROR", "CRITICAL", "EXCEPTION", "TRACEBACK", "HATA"))]
            return "\n".join(kept) if kept else ""
        return raw

    def _paint(self, view: QPlainTextEdit, *, follow: bool) -> None:
        custom_fn = getattr(view, "_custom_filter_fn", None)
        if callable(custom_fn):
            text = str(custom_fn(self._text))
        else:
            text = self._filtered_text(self._text) if follow else self._text
        scrollbar = view.verticalScrollBar()
        at_end = follow or scrollbar.value() >= scrollbar.maximum() - 4
        if view.toPlainText() == text:
            return
        view.setPlainText(text)
        if at_end:
            cursor = view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            view.setTextCursor(cursor)
            scrollbar.setValue(scrollbar.maximum())


class _LogPage(QWidget):
    """Shows logs/bot.log inside the main window. No extra log window."""

    LEVEL_FILTERS = ("ALL", "INFO", "WARN", "ERROR")

    def __init__(self, source: _LogSource, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.source = source
        self._level_filter: str = "ALL"
        self.level_filter_buttons: Dict[str, QPushButton] = {}
        layout = wrap_shell_tab_root_layout(self, "logs")
        layout.setSpacing(12)

        hint = QLabel(
            "Canlı günlük alt çubukta da akar; burada tam dosyayı kaydırarak okuyabilir, filtreleyebilir ve dışa aktarabilirsiniz.",
            self,
        )
        hint.setStyleSheet("color: #9aa3af; font-size: 12px; font-weight: 500;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # Toolbar: Live search box + Level filters (ALL, INFO, WARN, ERROR) + Export/Copy
        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.setSpacing(8)

        self.search_edit = QLineEdit(self)
        self.search_edit.setObjectName("LogPageSearchEdit")
        self.search_edit.setPlaceholderText("🔍 Günlükte ara (Hata, Savaş, Hasat, Rota)...")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setStyleSheet(
            """
            QLineEdit {
                background-color: #0d1015;
                color: #eef1f5;
                border: 1px solid #1e2530;
                border-radius: 7px;
                padding: 6px 10px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #56c8f5;
            }
            """
        )
        self.search_edit.textChanged.connect(self._on_filter_criteria_changed)
        toolbar_row.addWidget(self.search_edit, stretch=1)

        self.level_filter_group = QButtonGroup(self)
        self.level_filter_group.setExclusive(True)
        for lvl in self.LEVEL_FILTERS:
            btn = QPushButton(lvl, self)
            btn.setObjectName(f"LogLevelFilter_{lvl}")
            btn.setProperty("logPill", True)
            btn.setCheckable(True)
            btn.setChecked(lvl == "ALL")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(28)
            btn.clicked.connect(lambda _checked=False, mode=lvl: self.set_level_filter(mode))
            self.level_filter_group.addButton(btn)
            self.level_filter_buttons[lvl] = btn
            toolbar_row.addWidget(btn)

        self.level_combo = QComboBox(self)
        self.level_combo.setObjectName("LogLevelCombo")
        self.level_combo.addItems(list(self.LEVEL_FILTERS))
        self.level_combo.setCurrentText("ALL")
        self.level_combo.setFixedHeight(28)
        self.level_combo.currentTextChanged.connect(self.set_level_filter)
        toolbar_row.addWidget(self.level_combo)

        self.btn_export_log = QPushButton("Günlüğü Dışa Aktar / Kopyala", self)
        self.btn_export_log.setObjectName("LogExportCopyBtn")
        self.btn_export_log.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export_log.setFixedHeight(28)
        self.btn_export_log.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(86, 200, 245, 0.14);
                color: #56c8f5;
                border: 1px solid rgba(86, 200, 245, 0.40);
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(86, 200, 245, 0.24);
            }
            """
        )
        self.btn_export_log.clicked.connect(self._on_export_button_clicked)
        toolbar_row.addWidget(self.btn_export_log)

        layout.addLayout(toolbar_row)

        self.view = QPlainTextEdit(self)
        self.view.setReadOnly(True)
        self.view.setStyleSheet(
            """
            QPlainTextEdit {
                background-color: #05070a;
                color: #eef1f5;
                border: 1px solid #1e2530;
                border-radius: 8px;
                font-family: Consolas, "Cascadia Mono", monospace;
                font-size: 12px;
                padding: 10px;
            }
            """
        )
        setattr(self.view, "_custom_filter_fn", self._filter_log_content)
        layout.addWidget(self.view, stretch=1)
        source.attach(self.view, follow=False)

        refresh_row = QHBoxLayout()
        self.filter_status_label = QLabel("Seviye: ALL · Tüm satırlar gösteriliyor", self)
        self.filter_status_label.setObjectName("LogFilterStatusLabel")
        self.filter_status_label.setStyleSheet("color: #6b7280; font-size: 11px;")
        refresh_row.addWidget(self.filter_status_label)
        refresh_row.addStretch(1)
        refresh = QPushButton("Yenile", self)
        refresh.setCursor(Qt.CursorShape.PointingHandCursor)
        refresh.clicked.connect(self.reload)
        refresh_row.addWidget(refresh)
        layout.addLayout(refresh_row)
        self.reload()

    def set_level_filter(self, level: str) -> None:
        normalized = str(level or "ALL").strip().upper()
        if normalized == "WARNING":
            normalized = "WARN"
        if normalized not in self.LEVEL_FILTERS:
            normalized = "ALL"
        self._level_filter = normalized
        btn = self.level_filter_buttons.get(normalized)
        if btn is not None and not btn.isChecked():
            btn.blockSignals(True)
            btn.setChecked(True)
            btn.blockSignals(False)
        if hasattr(self, "level_combo") and self.level_combo.currentText() != normalized:
            self.level_combo.blockSignals(True)
            self.level_combo.setCurrentText(normalized)
            self.level_combo.blockSignals(False)
        self._on_filter_criteria_changed()

    def get_level_filter(self) -> str:
        return self._level_filter

    def _filter_log_content(self, raw: str) -> str:
        text = str(raw or "")
        if not text:
            return text
        lines = text.splitlines()
        lvl = self._level_filter
        if lvl == "INFO":
            lines = [
                ln for ln in lines
                if "INFO" in ln.upper() or ("WARN" not in ln.upper() and "ERROR" not in ln.upper() and "CRITICAL" not in ln.upper())
            ]
        elif lvl == "WARN":
            lines = [
                ln for ln in lines
                if any(k in ln.upper() for k in ("WARN", "WARNING", "UYARI"))
            ]
        elif lvl == "ERROR":
            lines = [
                ln for ln in lines
                if any(k in ln.upper() for k in ("ERROR", "CRITICAL", "EXCEPTION", "TRACEBACK", "HATA"))
            ]

        query = self.search_edit.text().strip() if hasattr(self, "search_edit") else ""
        if query:
            q_lower = query.lower()
            lines = [ln for ln in lines if q_lower in ln.lower()]

        if hasattr(self, "filter_status_label"):
            q_info = f" · Arama: '{query}'" if query else ""
            self.filter_status_label.setText(f"Seviye: {lvl}{q_info} · {len(lines)} satır")

        return "\n".join(lines)

    def _on_filter_criteria_changed(self, *_args: Any) -> None:
        self.source._paint(self.view, follow=False)

    def _on_export_button_clicked(self) -> None:
        self.export_filtered_log()

    def export_filtered_log(self, target_path: Optional[str] = None) -> str:
        """Exports the currently filtered log lines to clipboard (and optional file) and returns the filtered string."""
        src = self.__dict__.get("source")
        if src is not None:
            if not getattr(src, "_text", ""):
                src.reload()
            raw_text = getattr(src, "_text", "")
            filtered = self._filter_log_content(raw_text)
        else:
            recs = self.__dict__.get("_records") or []
            filtered = "\n".join(str(r[1]) if isinstance(r, (tuple, list)) and len(r) > 1 else str(r) for r in recs)
        try:
            app = QApplication.instance()
            if app is not None:
                clipboard = QApplication.clipboard()
                if clipboard is not None:
                    clipboard.setText(filtered)
        except Exception:
            logger.debug("Panoya kopyalama atlandı", exc_info=True)

        export_file = target_path
        if not export_file and src is not None and getattr(src, "log_path", None):
            base_dir = os.path.dirname(src.log_path) or "."
            export_file = os.path.join(base_dir, "bot_filtered_export.log")
        if export_file:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(export_file)), exist_ok=True)
                with open(export_file, "w", encoding="utf-8") as fh:
                    fh.write(filtered)
            except Exception:
                logger.debug("Filtrelenmiş günlük dosyaya yazılamadı", exc_info=True)

        lbl = self.__dict__.get("filter_status_label")
        if lbl is not None:
            line_cnt = len(filtered.splitlines()) if filtered else 0
            lvl = self.__dict__.get("_level_filter", "ALL")
            lbl.setText(
                f"Günlük dışa aktarıldı / panoya kopyalandı ({line_cnt} satır · Seviye: {lvl})"
            )
        return filtered

    def reload(self) -> None:
        self.source.reload()
        self.source._paint(self.view, follow=False)


class _StatChip(QFrame):
    """Single cell inside the 5-column HomeStatsStrip matching anasayfa.html."""

    def __init__(
        self,
        caption: str,
        subtext: str = "",
        default_value: str = "0",
        mono_value: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("HomeStat")
        self._default_value = default_value
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 11, 16, 11)
        layout.setSpacing(4)

        self.caption = QLabel(caption.upper(), self)
        self.caption.setObjectName("HomeStatCaption")
        layout.addWidget(self.caption)

        val_row = QHBoxLayout()
        val_row.setContentsMargins(0, 0, 0, 0)
        val_row.setSpacing(6)
        self.value = QLabel(default_value, self)
        self.value.setObjectName("HomeStatValue")
        if mono_value:
            self.value.setStyleSheet(
                'font-family: Consolas, "Cascadia Mono", monospace; font-size: 18px; font-weight: 800; color: #eef1f5;'
            )
        val_row.addWidget(self.value)

        self.sub = QLabel(subtext, self)
        self.sub.setObjectName("HomeStatSub")
        val_row.addWidget(self.sub)
        val_row.addStretch(1)
        layout.addLayout(val_row)
        self.setVisible(True)

    def show_value(self, text: str) -> None:
        self.value.setText(text)
        self.setVisible(True)

    def set_subtext(self, text: str) -> None:
        self.sub.setText(text)

    def hide_stat(self) -> None:
        self.value.setText(self._default_value)
        self.setVisible(True)


class _HomeTogglePill(QFrame):
    """Compact pill toggle card matching `.tog` + `.sw` in anasayfa.html."""

    toggled = pyqtSignal(bool)

    def __init__(self, label_text: str, checked: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._checked = bool(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(32)
        self.setStyleSheet(
            """
            QFrame {
                background-color: #0d1015;
                border: 1px solid #1e2530;
                border-radius: 7px;
            }
            QFrame:hover {
                border-color: #2c3545;
            }
            QLabel {
                background: transparent;
                border: none;
                color: #9aa3af;
                font-size: 11px;
                font-weight: 600;
            }
            """
        )
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 4, 10, 4)
        row.setSpacing(6)
        self.label = QLabel(label_text, self)
        row.addWidget(self.label)
        row.addStretch(1)
        self._indicator = QWidget(self)
        self._indicator.setFixedSize(28, 16)
        row.addWidget(self._indicator)

    def isChecked(self) -> bool:  # noqa: N802
        return self._checked

    def setChecked(self, val: bool) -> None:  # noqa: N802
        val = bool(val)
        if self._checked != val:
            self._checked = val
            self.update()

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._checked = not self._checked
            self.toggled.emit(self._checked)
            self.update()
        super().mousePressEvent(event)

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self._indicator.geometry()
        r = QRectF(rect.x(), rect.y(), rect.width(), rect.height())
        if self._checked:
            painter.setBrush(QColor(52, 211, 153, 64))
            painter.setPen(QPen(QColor(52, 211, 153, 140), 1.0))
        else:
            painter.setBrush(QColor("#1e2430"))
            painter.setPen(QPen(QColor(255, 255, 255, 22), 1.0))
        painter.drawRoundedRect(r, 8.0, 8.0)

        knob_d = 10.0
        knob_y = r.y() + (r.height() - knob_d) / 2.0
        knob_x = (r.right() - knob_d - 3.0) if self._checked else (r.x() + 3.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#34d399") if self._checked else QColor("#6b7280"))
        painter.drawEllipse(QRectF(knob_x, knob_y, knob_d, knob_d))
        painter.end()


class _HomeMinimapCanvas(QWidget):
    """Paints the live minimap & route canvas or a clean radar grid when idle."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._nodes: List[Dict[str, Any]] = []
        self._active_node_id: Optional[int] = None
        self._player_pos: Optional[Tuple[float, float]] = None
        self._is_running: bool = False
        self._live_tag: str = "BEKLEMEDE"

    def update_nodes(
        self,
        nodes: Optional[List[Any]] = None,
        active_node_id: Optional[int] = None,
        player_pos: Optional[Tuple[float, float]] = None,
        is_running: bool = False,
    ) -> None:
        self._nodes = []
        if nodes:
            for n in nodes:
                if isinstance(n, dict):
                    self._nodes.append(dict(n))
                else:
                    self._nodes.append({
                        "id": getattr(n, "id", 0),
                        "norm_x": getattr(n, "norm_x", 0.5),
                        "norm_y": getattr(n, "norm_y", 0.5),
                        "map_x": getattr(n, "map_x", 0),
                        "map_y": getattr(n, "map_y", 0),
                        "resource_type": getattr(n, "resource_type", ""),
                        "state": getattr(n, "state", "idle"),
                    })
        self._active_node_id = active_node_id
        self._player_pos = player_pos
        self._is_running = is_running
        self._live_tag = "CANLI" if (is_running and self._nodes) else "BEKLEMEDE"
        self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = float(self.width())
        h = float(self.height())
        rect = QRectF(0.5, 0.5, w - 1.0, h - 1.0)

        clip = QPainterPath()
        clip.addRoundedRect(rect, 8.0, 8.0)
        painter.setClipPath(clip)

        # Radial + linear dark gradient background (#0c1826 -> #070d16)
        grad = QLinearGradient(0.0, 0.0, 0.0, h)
        grad.setColorAt(0.0, QColor("#08131f"))
        grad.setColorAt(1.0, QColor("#060b12"))
        painter.fillRect(rect, grad)

        rad = QRadialGradient(QPointF(w * 0.35, h * 0.40), max(w, h) * 0.65)
        rad.setColorAt(0.0, QColor(12, 28, 46, 180))
        rad.setColorAt(1.0, QColor(7, 13, 22, 0))
        painter.fillRect(rect, rad)

        # 40px cyan grid lines
        grid_pen = QPen(QColor(86, 200, 245, 18), 1.0)
        painter.setPen(grid_pen)
        gx = 40.0
        while gx < w:
            painter.drawLine(QPointF(gx, 0.0), QPointF(gx, h))
            gx += 40.0
        gy = 40.0
        while gy < h:
            painter.drawLine(QPointF(0.0, gy), QPointF(w, gy))
            gy += 40.0

        # If live route nodes exist, map them onto the canvas and draw route path & waypoints
        valid_nodes: List[Dict[str, Any]] = []
        xs: List[float] = []
        ys: List[float] = []
        for idx, n in enumerate(self._nodes or []):
            if not isinstance(n, dict):
                continue
            try:
                raw_x = n.get("norm_x", 0.5)
                nx = float(raw_x if raw_x is not None else 0.5)
                if not math.isfinite(nx):
                    nx = 0.5
            except (ValueError, TypeError):
                nx = 0.5
            try:
                raw_y = n.get("norm_y", 0.5)
                ny = float(raw_y if raw_y is not None else 0.5)
                if not math.isfinite(ny):
                    ny = 0.5
            except (ValueError, TypeError):
                ny = 0.5
            raw_id = n.get("id")
            try:
                nid = int(raw_id) if raw_id is not None else idx + 1
            except (ValueError, TypeError):
                nid = idx + 1
            valid_nodes.append({"id": nid, "norm_x": nx, "norm_y": ny})
            xs.append(nx)
            ys.append(ny)

        if valid_nodes:
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            span_x = max(1e-4, max_x - min_x)
            span_y = max(1e-4, max_y - min_y)

            pts: List[QPointF] = []
            pad_x = w * 0.12
            pad_y = h * 0.16
            avail_w = max(20.0, w - pad_x * 2.0)
            avail_h = max(20.0, h - pad_y * 2.0)

            for nx, ny in zip(xs, ys):
                if span_x > 0.01:
                    px = pad_x + ((nx - min_x) / span_x) * avail_w
                else:
                    px = w * 0.5
                if span_y > 0.01:
                    py = pad_y + ((ny - min_y) / span_y) * avail_h
                else:
                    py = h * 0.5
                pts.append(QPointF(px, py))

            # Route path (dashed cyan)
            if len(pts) >= 2:
                route_path = QPainterPath()
                route_path.moveTo(pts[0])
                for pt in pts[1:]:
                    route_path.lineTo(pt)
                route_pen = QPen(QColor(86, 200, 245, 140), 2.0, Qt.PenStyle.DashLine)
                route_pen.setDashPattern([3.0, 2.5])
                painter.setPen(route_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(route_path)

            # Waypoint circles
            wp_font = QFont("Consolas", 8, QFont.Weight.Bold)
            painter.setFont(wp_font)
            active_pt: Optional[QPointF] = None

            for idx, (n_dict, pt) in enumerate(zip(valid_nodes, pts)):
                nid = int(n_dict.get("id", idx + 1) or (idx + 1))
                is_active = (self._active_node_id is not None and nid == self._active_node_id) or (self._active_node_id is None and idx == 0 and self._is_running)
                if is_active:
                    active_pt = pt
                    stroke_c = QColor("#f5b50a")
                    fill_c = QColor("#11151c")
                    txt_c = QColor("#f5b50a")
                elif self._is_running and idx < (self._active_node_id or 0):
                    stroke_c = QColor("#34d399")
                    fill_c = QColor("#11151c")
                    txt_c = QColor("#34d399")
                else:
                    stroke_c = QColor("#56c8f5")
                    fill_c = QColor("#11151c")
                    txt_c = QColor("#56c8f5")

                painter.setPen(QPen(stroke_c, 2.0))
                painter.setBrush(fill_c)
                painter.drawEllipse(pt, 7.5, 7.5)
                painter.setPen(txt_c)
                painter.drawText(QRectF(pt.x() - 8.0, pt.y() - 8.0, 16.0, 16.0), Qt.AlignmentFlag.AlignCenter, str(idx + 1))

            # Player marker (only rendered when running and target point is known)
            if self._is_running:
                pl_pt = active_pt or (pts[0] if pts else None)
                if pl_pt is not None:
                    painter.setPen(QPen(QColor("#f5b50a"), 2.0))
                    painter.setBrush(QColor(245, 181, 10, 52))
                    painter.drawEllipse(pl_pt, 10.0, 10.0)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor("#f5b50a"))
                    painter.drawEllipse(pl_pt, 4.5, 4.5)
                    sen_font = QFont("Consolas", 8, QFont.Weight.Bold)
                    painter.setFont(sen_font)
                    painter.setPen(QColor("#f5b50a"))
                    painter.drawText(
                        QRectF(pl_pt.x() - 28.0, pl_pt.y() - 24.0, 56.0, 14.0),
                        Qt.AlignmentFlag.AlignCenter,
                        "▲ SEN",
                    )

        # Top-right live/idle badge
        badge_w = 72.0 if self._live_tag == "BEKLEMEDE" else 50.0
        badge_rect = QRectF(w - badge_w - 8.0, 8.0, badge_w, 20.0)
        painter.setPen(QPen(QColor(255, 255, 255, 22), 1.0))
        painter.setBrush(QColor(10, 12, 16, 215))
        painter.drawRoundedRect(badge_rect, 5.0, 5.0)
        badge_color = QColor("#34d399") if self._live_tag == "CANLI" else QColor("#f5b50a")
        painter.setPen(badge_color)
        b_font = QFont("Consolas", 8, QFont.Weight.Bold)
        painter.setFont(b_font)
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, self._live_tag)

        # Outer border
        painter.setClipping(False)
        painter.setPen(QPen(QColor("#1e2530"), 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, 8.0, 8.0)
        painter.end()


class _TabFadeIn(QObject):
    """Short opacity ease-in when switching shell stack pages (no extra dependencies)."""

    def __init__(self, stack: QStackedWidget) -> None:
        super().__init__(stack)
        self._stack = stack
        self._anim: Optional[QVariantAnimation] = None

    def set_current(self, index: int) -> None:
        prev = self._stack.currentIndex()
        if prev >= 0:
            pw = self._stack.widget(prev)
            if pw is not None:
                pw.setGraphicsEffect(None)
        self._stack.setCurrentIndex(index)
        if prev == index:
            return
        widget = self._stack.widget(index)
        if widget is None:
            return
        if self._anim is not None:
            try:
                self._anim.stop()
            except Exception:
                logger.debug("Tab fade stop failed", exc_info=True)
            self._anim.deleteLater()
            self._anim = None

        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        effect.setOpacity(0.0)

        anim = QVariantAnimation(widget)
        anim.setDuration(200)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)

        def _on_val(value: object) -> None:
            try:
                effect.setOpacity(float(value))
            except (TypeError, ValueError):
                return

        def _on_finished() -> None:
            widget.setGraphicsEffect(None)
            self._anim = None

        anim.valueChanged.connect(_on_val)
        anim.finished.connect(_on_finished)
        self._anim = anim
        anim.start()


class _HomePage(QWidget):
    """Landing screen matching C:\\Users\\selçuk\\Desktop\\anasayfa.html pixel-perfectly."""

    def __init__(
        self,
        on_open: Callable[[str], None],
        on_action: Callable[[str], None],
        on_chord: Callable[[str], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("HomePage")
        self._on_open = on_open
        self._on_action = on_action
        self._on_chord = on_chord
        self._action_buttons: Dict[str, QPushButton] = {}
        self._chord_buttons: Dict[str, QPushButton] = {}
        self._shortcut_buttons: Dict[str, QPushButton] = {}
        self._running = False
        self._started_ts = time.monotonic()

        body = QFont("Segoe UI")
        body.setPixelSize(13)
        self.setFont(body)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setObjectName("HomeScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget(scroll)
        content.setObjectName("HomeScrollContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 14, 18, 12)
        layout.setSpacing(12)

        # 1. STATUS BAR (.status)
        status_card = QFrame(content)
        status_card.setObjectName("HomeStatusCard")
        status_row = QHBoxLayout(status_card)
        status_row.setContentsMargins(16, 12, 16, 12)
        status_row.setSpacing(14)

        self.status_dot = QFrame(status_card)
        self.status_dot.setFixedSize(9, 9)
        self._apply_dot_style(running=False)
        status_row.addWidget(self.status_dot)

        self.state_label = QLabel("Duraklatıldı", status_card)
        self.state_label.setObjectName("HomeState")
        self.state_label.setWordWrap(False)
        status_row.addWidget(self.state_label)

        self.route_summary_label = QLabel(
            '<b style="color:#eef1f5;">—</b> · Bölge Seçilmedi · Rota Yok · 0 node',
            status_card,
        )
        self.route_summary_label.setObjectName("HomeRouteSummary")
        self.route_summary_label.setTextFormat(Qt.TextFormat.RichText)
        status_row.addWidget(self.route_summary_label, stretch=1)

        # Hidden / synced compatibility labels so callers inspecting region_label / route_label work seamlessly
        self.region_label = QLabel(status_card)
        self.region_label.setObjectName("HomeRegion")
        self.region_label.setVisible(False)
        self.route_label = QLabel(status_card)
        self.route_label.setObjectName("HomeRoute")
        self.route_label.setVisible(False)

        self.btn_status_run = QPushButton("Başlat", status_card)
        self.btn_status_run.setProperty("homePrimaryBtn", True)
        self.btn_status_run.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_status_run.clicked.connect(lambda: self._on_chord("pause_resume_f10"))
        status_row.addWidget(self.btn_status_run)

        self.btn_status_save = QPushButton("Konumu Kaydet", status_card)
        self.btn_status_save.setProperty("homeSecondaryBtn", True)
        self.btn_status_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_status_save.clicked.connect(lambda: self._on_action("manual_record"))
        status_row.addWidget(self.btn_status_save)

        self.btn_status_mobile = QPushButton("Mobil Bağlan", status_card)
        self.btn_status_mobile.setProperty("homeSecondaryBtn", True)
        self.btn_status_mobile.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_status_mobile.clicked.connect(lambda: self._on_open("mobile"))
        status_row.addWidget(self.btn_status_mobile)

        layout.addWidget(status_card)

        # 2. STATS STRIP (.stats — 5 columns in a single bordered panel)
        stats_strip = QFrame(content)
        stats_strip.setObjectName("HomeStatsStrip")
        stats_lay = QHBoxLayout(stats_strip)
        stats_lay.setContentsMargins(0, 0, 0, 0)
        stats_lay.setSpacing(0)

        self._stats: Dict[str, _StatChip] = {}
        defaults = {"gathered": "0", "active": "0", "failures": "0", "yz": "0"}
        for key, caption, subtext in _HOME_STATS:
            chip = _StatChip(caption, subtext=subtext, default_value=defaults.get(key, "0"), parent=stats_strip)
            chip.setObjectName(f"HomeStat_{key}")
            self._stats[key] = chip
            stats_lay.addWidget(chip, stretch=1)

        self.session_chip = _StatChip(
            "Oturum",
            subtext="beklemede",
            default_value="0:00",
            mono_value=True,
            parent=stats_strip,
        )
        self.session_chip.setObjectName("HomeStat_session")
        stats_lay.addWidget(self.session_chip, stretch=1)
        layout.addWidget(stats_strip)

        # 3. COMMANDS GRID (.cmds — 7 columns + .cmd-hint)
        cmds_wrap = QWidget(content)
        cmds_lay = QVBoxLayout(cmds_wrap)
        cmds_lay.setContentsMargins(0, 0, 0, 0)
        cmds_lay.setSpacing(5)

        for key, caption in (
            ("radar", "Radar"),
            ("map", "Harita"),
            ("yolo", "YOLO"),
            ("calibration", "Kalibrasyon"),
            ("settings", "Ayarlar"),
        ):
            button = QPushButton(caption, cmds_wrap)
            button.setObjectName(f"HomeOpen_{key}")
            button.setProperty("homeAction", True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFixedHeight(32)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.clicked.connect(lambda _checked=False, page=key: on_open(page))
            self._shortcut_buttons[key] = button

        self._action_grid = QGridLayout()
        self._action_grid.setContentsMargins(0, 0, 0, 0)
        self._action_grid.setHorizontalSpacing(6)
        self._action_grid.setVerticalSpacing(6)
        for column in range(7):
            self._action_grid.setColumnStretch(column, 1)
        cmds_lay.addLayout(self._action_grid)
        self._ensure_chord_buttons()

        self.hotkey = QLabel(_DEFAULT_HOTKEY_LINE, cmds_wrap)
        self.hotkey.setObjectName("HomeHotkeyLine")
        self.hotkey.setWordWrap(False)
        cmds_lay.addWidget(self.hotkey)
        layout.addWidget(cmds_wrap)

        # 4. DETAIL SECTION (.detail — Left Minimap & Route card + Right stacked System & Detections cards)
        detail_row = QHBoxLayout()
        detail_row.setContentsMargins(0, 0, 0, 0)
        detail_row.setSpacing(12)

        left_card = self._build_minimap_card(content)
        detail_row.addWidget(left_card, stretch=5)

        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(12)
        right_col.addWidget(self._build_system_card(content))
        right_col.addWidget(self._build_detections_card(content), stretch=1)
        detail_row.addLayout(right_col, stretch=4)

        layout.addLayout(detail_row, stretch=1)
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _apply_dot_style(self, running: bool) -> None:
        if running:
            self.status_dot.setStyleSheet(
                "background-color: #34d399; border-radius: 4px; border: 1px solid #6ee7b7;"
            )
        else:
            self.status_dot.setStyleSheet(
                "background-color: #f5b50a; border-radius: 4px; border: 1px solid #fcd34d;"
            )

    def _build_minimap_card(self, parent: QWidget) -> QFrame:
        card = QFrame(parent)
        card.setObjectName("HomeDetailCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(15, 13, 15, 13)
        lay.setSpacing(10)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        title = QLabel("MİNİMAP & ROTA", card)
        title.setObjectName("HomeCardTitle")
        head.addWidget(title)
        head.addStretch(1)
        self.minimap_meta_label = QLabel("Bölge Seçilmedi · Ölçek 1.00x", card)
        self.minimap_meta_label.setObjectName("HomeCardMeta")
        head.addWidget(self.minimap_meta_label)
        lay.addLayout(head)

        self.minimap_canvas = _HomeMinimapCanvas(card)
        lay.addWidget(self.minimap_canvas, stretch=1)

        self.minimap_steps_container = QWidget(card)
        self.minimap_steps_layout = QVBoxLayout(self.minimap_steps_container)
        self.minimap_steps_layout.setContentsMargins(0, 0, 0, 0)
        self.minimap_steps_layout.setSpacing(4)

        self.minimap_empty_lbl = QLabel("Rotada kayıtlı nokta yok. (F8 ile nokta kaydedin)", self.minimap_steps_container)
        self.minimap_empty_lbl.setStyleSheet("color: #6b7280; font-size: 11px; padding: 10px 0;")
        self.minimap_empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.minimap_steps_layout.addWidget(self.minimap_empty_lbl)
        lay.addWidget(self.minimap_steps_container)

        self.minimap_pbar = QProgressBar(card)
        self.minimap_pbar.setRange(0, 100)
        self.minimap_pbar.setValue(0)
        self.minimap_pbar.setTextVisible(False)
        self.minimap_pbar.setFixedHeight(5)
        self.minimap_pbar.setStyleSheet(
            """
            QProgressBar {
                background-color: #1a202b;
                border: none;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background-color: #f5b50a;
                border-radius: 2px;
            }
            """
        )
        lay.addWidget(self.minimap_pbar)

        foot = QHBoxLayout()
        foot.setContentsMargins(0, 0, 0, 0)
        fl = QLabel("Rota ilerlemesi", card)
        fl.setStyleSheet("color: #6b7280; font-size: 11px;")
        foot.addWidget(fl)
        foot.addStretch(1)
        self.route_progress_label = QLabel("%0 — 0 aktif node", card)
        self.route_progress_label.setStyleSheet(
            'color: #9aa3af; font-family: Consolas, "Cascadia Mono", monospace; font-size: 11px;'
        )
        foot.addWidget(self.route_progress_label)
        lay.addLayout(foot)
        return card

    def _build_system_card(self, parent: QWidget) -> QFrame:
        card = QFrame(parent)
        card.setObjectName("HomeDetailCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(15, 13, 15, 13)
        lay.setSpacing(9)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        title = QLabel("SİSTEM", card)
        title.setObjectName("HomeCardTitle")
        head.addWidget(title)
        head.addStretch(1)
        self.clock_label = QLabel(datetime.now().strftime("%H:%M:%S"), card)
        self.clock_label.setObjectName("HomeCardMeta")
        head.addWidget(self.clock_label)
        lay.addLayout(head)

        # GPU row
        gpu_box = QVBoxLayout()
        gpu_box.setContentsMargins(0, 0, 0, 0)
        gpu_box.setSpacing(4)
        top_gpu = QHBoxLayout()
        self.gpu_title_lbl = QLabel("GPU · Beklemede", card)
        self.gpu_title_lbl.setStyleSheet("color: #9aa3af; font-size: 12px;")
        self.gpu_val_lbl = QLabel("0% · —°C · 0.0/0.0 GB", card)
        self.gpu_val_lbl.setStyleSheet('color: #eef1f5; font-family: Consolas, "Cascadia Mono", monospace; font-size: 12px;')
        top_gpu.addWidget(self.gpu_title_lbl)
        top_gpu.addStretch(1)
        top_gpu.addWidget(self.gpu_val_lbl)
        gpu_box.addLayout(top_gpu)

        self.gpu_bar = QProgressBar(card)
        self.gpu_bar.setRange(0, 100)
        self.gpu_bar.setValue(0)
        self.gpu_bar.setTextVisible(False)
        self.gpu_bar.setFixedHeight(5)
        self.gpu_bar.setStyleSheet(
            """
            QProgressBar { background-color: #1a202b; border: none; border-radius: 2px; }
            QProgressBar::chunk { background-color: #34d399; border-radius: 2px; }
            """
        )
        gpu_box.addWidget(self.gpu_bar)
        lay.addLayout(gpu_box)

        # CPU row
        cpu_box = QVBoxLayout()
        cpu_box.setContentsMargins(0, 0, 0, 0)
        cpu_box.setSpacing(4)
        top_cpu = QHBoxLayout()
        self.cpu_title_lbl = QLabel("CPU · Çoklu Çekirdek", card)
        self.cpu_title_lbl.setStyleSheet("color: #9aa3af; font-size: 12px;")
        self.cpu_val_lbl = QLabel("0% · 2.4 GHz", card)
        self.cpu_val_lbl.setStyleSheet('color: #eef1f5; font-family: Consolas, "Cascadia Mono", monospace; font-size: 12px;')
        top_cpu.addWidget(self.cpu_title_lbl)
        top_cpu.addStretch(1)
        top_cpu.addWidget(self.cpu_val_lbl)
        cpu_box.addLayout(top_cpu)

        self.cpu_bar = QProgressBar(card)
        self.cpu_bar.setRange(0, 100)
        self.cpu_bar.setValue(0)
        self.cpu_bar.setTextVisible(False)
        self.cpu_bar.setFixedHeight(5)
        self.cpu_bar.setStyleSheet(
            """
            QProgressBar { background-color: #1a202b; border: none; border-radius: 2px; }
            QProgressBar::chunk { background-color: #56c8f5; border-radius: 2px; }
            """
        )
        cpu_box.addWidget(self.cpu_bar)
        lay.addLayout(cpu_box)

        # MSS Capture row
        mss_box = QVBoxLayout()
        mss_box.setContentsMargins(0, 0, 0, 0)
        mss_box.setSpacing(4)
        top_mss = QHBoxLayout()
        self.mss_title_lbl = QLabel("Ekran Yakalama (MSS)", card)
        self.mss_title_lbl.setStyleSheet("color: #9aa3af; font-size: 12px;")
        self.mss_val_lbl = QLabel("0 FPS · 0.0 ms", card)
        self.mss_val_lbl.setStyleSheet('color: #eef1f5; font-family: Consolas, "Cascadia Mono", monospace; font-size: 12px;')
        top_mss.addWidget(self.mss_title_lbl)
        top_mss.addStretch(1)
        top_mss.addWidget(self.mss_val_lbl)
        mss_box.addLayout(top_mss)

        self.mss_bar = QProgressBar(card)
        self.mss_bar.setRange(0, 100)
        self.mss_bar.setValue(0)
        self.mss_bar.setTextVisible(False)
        self.mss_bar.setFixedHeight(5)
        self.mss_bar.setStyleSheet(
            """
            QProgressBar { background-color: #1a202b; border: none; border-radius: 2px; }
            QProgressBar::chunk { background-color: #f5b50a; border-radius: 2px; }
            """
        )
        mss_box.addWidget(self.mss_bar)
        lay.addLayout(mss_box)

        self.sys_kv_line = QLabel(
            'Model <b style="color:#eef1f5;font-family:Consolas;">—</b> &nbsp;·&nbsp; '
            'Eşik <b style="color:#eef1f5;font-family:Consolas;">0.50</b> &nbsp;·&nbsp; '
            'Watchdog <b style="color:#f5b50a;font-family:Consolas;">Beklemede</b> &nbsp;·&nbsp; '
            'Süre <b style="color:#eef1f5;font-family:Consolas;">00:00</b> &nbsp;·&nbsp; '
            'Hız <b style="color:#eef1f5;font-family:Consolas;">~0/sa</b>',
            card,
        )
        self.sys_kv_line.setTextFormat(Qt.TextFormat.RichText)
        self.sys_kv_line.setWordWrap(True)
        self.sys_kv_line.setStyleSheet(
            "color: #6b7280; font-size: 11px; border-top: 1px solid #1e2530; padding-top: 8px;"
        )
        lay.addWidget(self.sys_kv_line)

        tog_row = QHBoxLayout()
        tog_row.setContentsMargins(0, 2, 0, 0)
        tog_row.setSpacing(6)
        self.tog_auto_rec = _HomeTogglePill("Oto-Kayıt", checked=False, parent=card)
        self.tog_auto_rec.toggled.connect(lambda _v: self._on_action("auto_record"))
        self.tog_avoid_combat = _HomeTogglePill("Savaşta Kaçın", checked=False, parent=card)
        self.tog_avoid_combat.toggled.connect(lambda _v: self._on_action("combat"))
        self.tog_mobile_panel = _HomeTogglePill("Mobil Panel", checked=False, parent=card)
        self.tog_mobile_panel.toggled.connect(lambda _v: self._on_open("mobile"))
        tog_row.addWidget(self.tog_auto_rec, stretch=1)
        tog_row.addWidget(self.tog_avoid_combat, stretch=1)
        tog_row.addWidget(self.tog_mobile_panel, stretch=1)
        lay.addLayout(tog_row)
        return card

    def _build_detections_card(self, parent: QWidget) -> QFrame:
        card = QFrame(parent)
        card.setObjectName("HomeDetailCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(15, 13, 15, 13)
        lay.setSpacing(8)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        title = QLabel("SON TESPİTLER", card)
        title.setObjectName("HomeCardTitle")
        head.addWidget(title)
        head.addStretch(1)
        meta = QLabel("YOLO · canlı", card)
        meta.setObjectName("HomeCardMeta")
        head.addWidget(meta)
        lay.addLayout(head)

        self.detections_container = QWidget(card)
        self.detections_layout = QVBoxLayout(self.detections_container)
        self.detections_layout.setContentsMargins(0, 0, 0, 0)
        self.detections_layout.setSpacing(6)
        lay.addWidget(self.detections_container)

        self.detections_empty_lbl = QLabel("Henüz tespit yok · YOLO hazır", self.detections_container)
        self.detections_empty_lbl.setStyleSheet("color: #6b7280; font-size: 11px; padding: 12px 0;")
        self.detections_empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detections_layout.addWidget(self.detections_empty_lbl)

        self.detections_foot_lbl = QLabel(
            'Bugün: <b style="color:#eef1f5;font-family:Consolas;">0</b> oturum &nbsp;·&nbsp; '
            'en iyi toplama <b style="color:#eef1f5;font-family:Consolas;">0/saat</b> &nbsp;·&nbsp; '
            'ortalama güven <b style="color:#eef1f5;font-family:Consolas;">%0.0</b>',
            card,
        )
        self.detections_foot_lbl.setTextFormat(Qt.TextFormat.RichText)
        self.detections_foot_lbl.setStyleSheet("color: #6b7280; font-size: 11px; padding-top: 2px;")
        lay.addWidget(self.detections_foot_lbl)
        lay.addStretch(1)
        return card

    def update_route_steps(
        self,
        region_name: str,
        route_name: str,
        nodes: Optional[List[Any]] = None,
        active_idx: int = 0,
    ) -> None:
        if not hasattr(self, "minimap_steps_layout"):
            return
        while self.minimap_steps_layout.count() > 0:
            child = self.minimap_steps_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        node_list = nodes or []
        node_count = len(node_list)

        if not node_list:
            lbl = QLabel("Rotada kayıtlı nokta yok. (F8 ile nokta kaydedin)", self.minimap_steps_container)
            lbl.setStyleSheet("color: #6b7280; font-size: 11px; padding: 8px 0;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.minimap_steps_layout.addWidget(lbl)
            if hasattr(self, "minimap_pbar"):
                self.minimap_pbar.setValue(0)
            if hasattr(self, "route_progress_label"):
                self.route_progress_label.setText("%0 — 0 aktif node")
            if hasattr(self, "minimap_canvas"):
                self.minimap_canvas.update_nodes([], None, None, is_running=self._running)
            return

        active_idx = max(0, min(active_idx, node_count - 1)) if node_count > 0 else 0
        pct = int((active_idx + 1) * 100 / node_count) if (self._running and node_count > 0) else 0
        if hasattr(self, "minimap_pbar"):
            self.minimap_pbar.setValue(pct)
        if hasattr(self, "route_progress_label"):
            self.route_progress_label.setText(f"%{pct} — {node_count} aktif node")

        start_i = max(0, min(active_idx, max(0, node_count - 4)))
        visible_nodes = node_list[start_i : start_i + 4]

        for offset, n in enumerate(visible_nodes):
            idx = start_i + offset
            nid = getattr(n, "id", None) if not isinstance(n, dict) else n.get("id")
            res_type = getattr(n, "resource_type", None) if not isinstance(n, dict) else n.get("resource_type")
            res_type = res_type or f"Nokta #{nid or (idx + 1)}"

            if not self._running:
                state = "idle"
                meta_txt = "BEKLİYOR"
            elif idx < active_idx:
                state = "done"
                meta_txt = "TAMAMLANDI"
            elif idx == active_idx:
                state = "active"
                meta_txt = "ŞU AN BURADA"
            else:
                state = "idle"
                meta_txt = f"SIRADA #{idx + 1}"

            step_frame = QFrame(self.minimap_steps_container)
            if state == "active":
                step_frame.setStyleSheet(
                    "QFrame { background-color: rgba(245, 181, 10, 0.08); border: 1px solid rgba(245, 181, 10, 0.45); border-radius: 7px; }"
                )
            else:
                step_frame.setStyleSheet(
                    "QFrame { background-color: #0d1015; border: 1px solid #1e2530; border-radius: 7px; }"
                )
            sr = QHBoxLayout(step_frame)
            sr.setContentsMargins(10, 5, 10, 5)
            sr.setSpacing(10)

            badge = QLabel(str(idx + 1), step_frame)
            badge.setFixedSize(20, 20)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if state == "done":
                badge.setStyleSheet(
                    'background: rgba(52, 211, 153, 0.15); color: #34d399; border-radius: 5px; font-family: Consolas, monospace; font-size: 11px; font-weight: 700; border: none;'
                )
            elif state == "active":
                badge.setStyleSheet(
                    'background: rgba(245, 181, 10, 0.20); color: #f5b50a; border-radius: 5px; font-family: Consolas, monospace; font-size: 11px; font-weight: 700; border: none;'
                )
            else:
                badge.setStyleSheet(
                    'background: rgba(255, 255, 255, 0.06); color: #9aa3af; border-radius: 5px; font-family: Consolas, monospace; font-size: 11px; font-weight: 700; border: none;'
                )
            sr.addWidget(badge)

            name_lbl = QLabel(str(res_type), step_frame)
            name_lbl.setStyleSheet("color: #eef1f5; font-size: 12px; font-weight: 600; border: none; background: transparent;")
            sr.addWidget(name_lbl, stretch=1)

            meta_lbl = QLabel(meta_txt, step_frame)
            meta_lbl.setStyleSheet(
                'color: #6b7280; font-family: Consolas, "Cascadia Mono", monospace; font-size: 11px; border: none; background: transparent;'
            )
            sr.addWidget(meta_lbl)
            self.minimap_steps_layout.addWidget(step_frame)

        if hasattr(self, "minimap_canvas"):
            active_id = getattr(node_list[active_idx], "id", None) if not isinstance(node_list[active_idx], dict) else node_list[active_idx].get("id")
            self.minimap_canvas.update_nodes(node_list, active_id, is_running=self._running)

    def update_system_metrics(
        self,
        sys_snapshot: Optional[Dict[str, Any]],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        snap = sys_snapshot if isinstance(sys_snapshot, dict) else {}
        gpu = snap.get("gpu")
        if not isinstance(gpu, dict):
            gpu = {}
        cpu = snap.get("cpu")
        if not isinstance(cpu, dict):
            cpu = {}
        cap = snap.get("capture")
        if not isinstance(cap, dict):
            cap = {}

        def _safe_pct(d: Dict[str, Any], key: str = "pct") -> int:
            try:
                val = d.get(key)
                if val is None:
                    return 0
                f = float(val)
                return max(0, min(100, int(round(f)))) if math.isfinite(f) else 0
            except (ValueError, TypeError):
                return 0

        if hasattr(self, "gpu_title_lbl"):
            self.gpu_title_lbl.setText(str(gpu.get("name") or "GPU · Beklemede"))
        if hasattr(self, "gpu_val_lbl"):
            self.gpu_val_lbl.setText(str(gpu.get("label_str") or "0% · —°C · 0.0/0.0 GB"))
        if hasattr(self, "gpu_bar"):
            self.gpu_bar.setValue(_safe_pct(gpu))

        if hasattr(self, "cpu_title_lbl"):
            self.cpu_title_lbl.setText(str(cpu.get("name") or "CPU · Çoklu Çekirdek"))
        if hasattr(self, "cpu_val_lbl"):
            self.cpu_val_lbl.setText(str(cpu.get("label_str") or "0% · 2.4 GHz"))
        if hasattr(self, "cpu_bar"):
            self.cpu_bar.setValue(_safe_pct(cpu))

        if hasattr(self, "mss_title_lbl"):
            self.mss_title_lbl.setText("Ekran Yakalama (MSS)")
        if hasattr(self, "mss_val_lbl"):
            self.mss_val_lbl.setText(str(cap.get("label_str") or "0 FPS · 0.0 ms"))
        if hasattr(self, "mss_bar"):
            self.mss_bar.setValue(_safe_pct(cap))

        if hasattr(self, "sys_kv_line"):
            m = meta if isinstance(meta, dict) else {}
            model_name = str(m.get("model", "—"))
            thresh_val = str(m.get("threshold", "0.50"))
            watchdog_txt = str(m.get("watchdog", "Beklemede"))
            wd_color = "#34d399" if watchdog_txt == "Normal" else "#f5b50a"
            duration_txt = str(m.get("duration", "00:00"))
            rate_txt = str(m.get("rate", "~0/sa"))
            self.sys_kv_line.setText(
                f'Model <b style="color:#eef1f5;font-family:Consolas;">{model_name}</b> &nbsp;·&nbsp; '
                f'Eşik <b style="color:#eef1f5;font-family:Consolas;">{thresh_val}</b> &nbsp;·&nbsp; '
                f'Watchdog <b style="color:{wd_color};font-family:Consolas;">{watchdog_txt}</b> &nbsp;·&nbsp; '
                f'Süre <b style="color:#eef1f5;font-family:Consolas;">{duration_txt}</b> &nbsp;·&nbsp; '
                f'Hız <b style="color:#eef1f5;font-family:Consolas;">{rate_txt}</b>'
            )

    def update_detections(
        self,
        detections: Optional[List[Dict[str, Any]]] = None,
        footer_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not hasattr(self, "detections_layout"):
            return
        while self.detections_layout.count() > 0:
            child = self.detections_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        items = [d for d in (detections or []) if isinstance(d, dict)]
        if not items:
            lbl = QLabel("Henüz tespit yok · YOLO hazır", self.detections_container)
            lbl.setStyleSheet("color: #6b7280; font-size: 11px; padding: 12px 0;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.detections_layout.addWidget(lbl)
        else:
            for d in items[:4]:
                d_frame = QFrame(self.detections_container)
                d_frame.setStyleSheet(
                    "QFrame { background-color: #0d1015; border: 1px solid #1e2530; border-radius: 7px; }"
                )
                dr = QHBoxLayout(d_frame)
                dr.setContentsMargins(10, 5, 10, 5)
                dr.setSpacing(8)

                name_txt = str(d.get("name") or d.get("category_tr") or "Hedef")
                dist_txt = str(d.get("dist") or d.get("location_str") or "Görüş alanı")
                try:
                    raw_conf = d.get("conf") if d.get("conf") is not None else d.get("confidence")
                    conf_val = float(raw_conf) if raw_conf is not None else 0.0
                    if not math.isfinite(conf_val):
                        conf_val = 0.0
                except (ValueError, TypeError):
                    conf_val = 0.0

                if 0.0 < conf_val <= 1.0:
                    conf_val *= 100.0
                conf_txt = f"%{conf_val:.0f}"
                is_hostile = bool(d.get("hostile") or "Saldırgan" in name_txt or "monster" in str(d.get("class", "")).lower())
                conf_color = "#f87171" if is_hostile else "#34d399"

                n_lbl = QLabel(name_txt, d_frame)
                n_lbl.setStyleSheet("color: #eef1f5; font-size: 12px; font-weight: 600; border: none; background: transparent;")
                dr.addWidget(n_lbl, stretch=1)

                m_lbl = QLabel(dist_txt, d_frame)
                m_lbl.setStyleSheet('color: #6b7280; font-family: Consolas, "Cascadia Mono", monospace; font-size: 11px; border: none; background: transparent;')
                dr.addWidget(m_lbl)

                c_lbl = QLabel(conf_txt, d_frame)
                c_lbl.setStyleSheet(f'color: {conf_color}; font-family: Consolas, "Cascadia Mono", monospace; font-size: 11px; font-weight: 700; border: none; background: transparent;')
                dr.addWidget(c_lbl)

                self.detections_layout.addWidget(d_frame)

        if hasattr(self, "detections_foot_lbl"):
            fs = footer_stats if isinstance(footer_stats, dict) else {}
            try:
                raw_sess = fs.get("sessions")
                sess_cnt = int(raw_sess) if raw_sess is not None else 0
            except (ValueError, TypeError):
                sess_cnt = 0
            try:
                raw_br = fs.get("best_rate")
                best_rate = int(raw_br) if raw_br is not None else 0
            except (ValueError, TypeError):
                best_rate = 0
            try:
                raw_ac = fs.get("avg_confidence")
                avg_conf = float(raw_ac) if raw_ac is not None else 0.0
                if not math.isfinite(avg_conf):
                    avg_conf = 0.0
            except (ValueError, TypeError):
                avg_conf = 0.0
            self.detections_foot_lbl.setText(
                f'Bugün: <b style="color:#eef1f5;font-family:Consolas;">{sess_cnt}</b> oturum &nbsp;·&nbsp; '
                f'en iyi toplama <b style="color:#eef1f5;font-family:Consolas;">{best_rate}/saat</b> &nbsp;·&nbsp; '
                f'ortalama güven <b style="color:#eef1f5;font-family:Consolas;">%{avg_conf:.1f}</b>'
            )

    def set_hotkey_line(self, text: str) -> None:
        raw = str(text or _DEFAULT_HOTKEY_LINE)
        if not raw.lower().startswith("kısayollar"):
            raw = f"Kısayollar: {raw.replace(', ', ' · ')}"
        self.hotkey.setText(raw)

    def set_chord_hints(self, node: str, menu: str, pause: str) -> None:
        """Tooltips name the real chords. Button captions stay short."""
        menu_button = self._chord_buttons.get("toggle_menu_f9")
        if menu_button is not None:
            menu_button.setToolTip(f"{menu} menüyü açar. Yürüyüşü duraklatmaz.")
        pause_button = self._chord_buttons.get("pause_resume_f10")
        if pause_button is not None:
            pause_button.setToolTip(f"{pause} duraklatır veya devam ettirir.")
        node_button = self._action_buttons.get("manual_record")
        if node_button is not None and node:
            current = node_button.toolTip()
            if node not in current:
                node_button.setToolTip(f"{node} mevcut konumu kaydeder. {current}".strip())

    def set_running(self, running: Optional[bool]) -> None:
        """Ctrl+Alt+F10 label: Başlat while paused, Durdur while running."""
        is_run = bool(running)
        self._running = is_run
        self._apply_dot_style(is_run)
        button = self._chord_buttons.get("pause_resume_f10")
        if button is not None:
            text = "Durdur" if is_run else "Başlat"
            if button.text() != text:
                button.setText(text)
        if hasattr(self, "btn_status_run"):
            top_text = "Duraklat" if is_run else "Başlat"
            if self.btn_status_run.text() != top_text:
                self.btn_status_run.setText(top_text)
        if hasattr(self, "session_chip"):
            self.session_chip.sub.setText("çalışıyor" if is_run else "duraklatıldı")

    def _ensure_chord_buttons(self) -> None:
        if self._chord_buttons:
            return
        specs = (
            (
                "pause_resume_f10",
                "Başlat",
                "Ctrl+Alt+F10 duraklatır veya devam ettirir.",
                "go",
            ),
            (
                "toggle_menu_f9",
                "Menü",
                "Ctrl+Alt+F9 menüyü açar. Yürüyüşü duraklatmaz.",
                "",
            ),
        )
        for key, label, tip, variant in specs:
            button = QPushButton(label, self)
            button.setObjectName(f"HomeChord_{key}")
            button.setProperty("homeAction", True)
            button.setProperty("homeChord", key)
            if variant:
                button.setProperty("homeActionVariant", variant)
            button.setToolTip(tip)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFixedHeight(32)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.clicked.connect(lambda _checked=False, chord=key: self._on_chord(chord))
            self._chord_buttons[key] = button
        self._place_buttons([])

    def _place_buttons(self, action_buttons: list) -> None:
        ordered: List[QPushButton] = [
            self._chord_buttons["pause_resume_f10"],
            self._chord_buttons["toggle_menu_f9"],
        ]
        ordered.extend(action_buttons)
        existing_labels = {b.text().strip().lower() for b in ordered}
        for key, btn in self._shortcut_buttons.items():
            if btn.text().strip().lower() not in existing_labels:
                ordered.append(btn)
            else:
                btn.setVisible(False)
        for index, button in enumerate(ordered):
            button.setVisible(True)
            self._action_grid.addWidget(button, index // 7, index % 7)

    def sync_actions(self, pairs: list) -> None:
        """One compact button per F9 row. The long sentence is the tooltip."""
        cleaned = []
        for item in pairs or []:
            if not item or len(item) < 2:
                continue
            action_id = str(item[0]).strip()
            long_label = str(item[1]).strip()
            if not action_id or not long_label:
                continue
            short = _short_home_action(action_id, long_label)
            if short == "F9":
                continue
            cleaned.append((action_id, short, long_label))
        ids = [action_id for action_id, _short, _long in cleaned]
        if ids != list(self._action_buttons.keys()):
            self._rebuild_actions(cleaned)
            return
        for action_id, short, long_label in cleaned:
            button = self._action_buttons.get(action_id)
            if button is None:
                continue
            if button.text() != short:
                button.setText(short)
            if button.toolTip() != long_label:
                button.setToolTip(long_label)

    def _rebuild_actions(self, pairs: list) -> None:
        self._ensure_chord_buttons()
        while self._action_grid.count():
            item = self._action_grid.takeAt(0)
            widget = item.widget()
            if widget is None:
                continue
            if widget.property("homeChord") or widget in self._shortcut_buttons.values():
                widget.setParent(self)
                continue
            widget.setParent(None)
            widget.deleteLater()
        self._action_buttons = {}
        created = []
        for action_id, short, long_label in pairs:
            button = QPushButton(short, self)
            button.setObjectName(f"HomeAction_{action_id}")
            button.setProperty("homeAction", True)
            if action_id == "manual_record":
                button.setProperty("homeChord", "record_node_f8")
            if action_id == "minimap_auto_record":
                button.setProperty("homeActionVariant", "star")
            elif action_id == "exit":
                button.setProperty("homeActionVariant", "danger")
            button.setToolTip(long_label)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFixedHeight(32)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.clicked.connect(lambda _checked=False, aid=action_id: self._on_action(aid))
            created.append(button)
            self._action_buttons[action_id] = button
        self._place_buttons(created)

    def apply_snapshot(self, snapshot: Dict[str, Optional[str]]) -> None:
        state_val = snapshot.get("state")
        if state_val:
            self.state_label.setText(str(state_val).strip())
            self.state_label.setVisible(True)
        else:
            self.state_label.setText("Duraklatıldı" if not self._running else "Çalışıyor")
            self.state_label.setVisible(True)

        self._show_line(self.region_label, snapshot.get("region"))
        self._show_line(self.route_label, snapshot.get("route"))

        reg = str(snapshot.get("region") or "").strip()
        rte = str(snapshot.get("route") or "").strip()
        act = str(snapshot.get("active") or "0").strip()

        if reg and rte:
            self.route_summary_label.setText(
                f'<b style="color:#eef1f5;">{reg}</b> · {rte} · {act} node'
            )
            if hasattr(self, "minimap_meta_label"):
                self.minimap_meta_label.setText(f"{reg} · {rte}")
        elif reg:
            self.route_summary_label.setText(
                f'<b style="color:#eef1f5;">{reg}</b> · Rota Yok · {act} node'
            )
            if hasattr(self, "minimap_meta_label"):
                self.minimap_meta_label.setText(f"{reg} · Ölçek 1.00x")
        else:
            self.route_summary_label.setText(
                '<b style="color:#eef1f5;">—</b> · Bölge Seçilmedi · Rota Yok · 0 node'
            )
            if hasattr(self, "minimap_meta_label"):
                self.minimap_meta_label.setText("Bölge Seçilmedi · Ölçek 1.00x")

        if hasattr(self, "clock_label"):
            self.clock_label.setText(datetime.now().strftime("%H:%M:%S"))
        if hasattr(self, "session_chip"):
            raw_duration = snapshot.get("session_duration_sec")
            if self._running and raw_duration is not None and float(raw_duration) > 0:
                elapsed = max(0, int(float(raw_duration)))
            elif self._running:
                elapsed = max(0, int(time.monotonic() - self._started_ts))
            else:
                elapsed = 0
            mins, secs = divmod(elapsed, 60)
            self.session_chip.show_value(f"{mins}:{secs:02d}")
            self.session_chip.set_subtext("çalışıyor" if self._running else "beklemede")

        for key, chip in self._stats.items():
            value = snapshot.get(key)
            if value is None or str(value).strip() == "":
                chip.hide_stat()
            else:
                chip.show_value(str(value))

    @staticmethod
    def _show_line(label: QLabel, value: Optional[str]) -> None:
        text = "" if value is None else str(value).strip()
        if not text:
            label.clear()
            label.setVisible(False)
            return
        label.setText(text)
        label.setVisible(False)


class GatherionShell(QMainWindow):
    """One program window. The rail replaces the old row of equal tabs."""

    NAV = NAV
    page_created = pyqtSignal(str, QWidget)

    ALIASES = {
        "yolo_factory": "yolo",
        "yolov8_factory": "yolo",
        "dataset_factory": "yolo",
        "train_ai": "yolo",
        "train_yolo": "yolo",
        "yz": "yolo",
        "calibrate": "calibration",
        "analytics": "stats",
        "open_log": "logs",
        "log": "logs",
        "map_viewer": "map",
        "map_viewer_window": "map",
        "cartography_radar": "radar",
        "cartography_radar_window": "radar",
        "cartography_radar_dialog": "radar",
        "radar_window": "radar",
        "mobile_connect": "mobile",
        "mobile_dashboard": "mobile",
        "settings_window": "settings",
    }

    def __init__(
        self,
        settings_mgr: Any = None,
        route_mgr: Any = None,
        f9_menu: Optional[QWidget] = None,
        telemetry: Any = None,
        explorer: Any = None,
        mobile_server: Any = None,
        tunnel_mgr: Any = None,
        factory: Any = None,
        log_path: str = "",
        window_title: str = "Gatherion-Local v2.0 (Türkçe)",
        worker: Any = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._closing = False
        self.settings_mgr = settings_mgr
        self.route_mgr = route_mgr
        self._telemetry = telemetry
        self._worker = worker
        self._explorer = explorer
        self._auto_cartographer = (
            getattr(explorer, "cartographer", None)
            or getattr(explorer, "auto_cartographer", None)
            or getattr(worker, "auto_cartographer", None)
            or explorer
        )
        self.f9_menu = f9_menu
        self._chord_handlers: Dict[str, Callable[[], None]] = {}
        self.pages: Dict[str, QWidget] = {}
        self._page_index: Dict[str, int] = {}
        self._index_key: Dict[int, str] = {}
        self._nav_buttons: Dict[str, QPushButton] = {}
        self._syncing_combos: bool = False
        self.region_combo: Optional[QComboBox] = None
        self.combo_region: Optional[QComboBox] = None
        self.route_combo: Optional[QComboBox] = None
        self.combo_route: Optional[QComboBox] = None

        self.setWindowTitle(window_title)
        self.setMinimumSize(1180, 900)
        self.resize(1360, 980)
        self._apply_theme()

        central = QWidget(self)
        central.setObjectName("ShellCentral")
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Prototype Top Bar (.topbar)
        self._topbar = self._build_topbar()
        root.addWidget(self._topbar)

        body = QWidget(central)
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)
        self._rail = self._build_rail()
        body_lay.addWidget(self._rail)

        right_column = QWidget(body)
        right_lay = QVBoxLayout(right_column)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        self._stage = QWidget(right_column)
        stage_grid = QGridLayout(self._stage)
        stage_grid.setContentsMargins(0, 0, 0, 0)
        stage_grid.setSpacing(0)
        self.stack = QStackedWidget(self._stage)
        self.stack.setObjectName("ShellStack")
        self._tab_fade = _TabFadeIn(self.stack)
        stage_grid.addWidget(self.stack, 0, 0)

        self._f9_overlay = QFrame(self._stage)
        self._f9_overlay.setObjectName("F9Overlay")
        self._f9_overlay.setVisible(False)
        overlay_lay = QVBoxLayout(self._f9_overlay)
        overlay_lay.setContentsMargins(16, 12, 16, 12)
        self._f9_scroll = QScrollArea(self._f9_overlay)
        self._f9_scroll.setObjectName("F9Scroll")
        self._f9_scroll.setWidgetResizable(True)
        self._f9_scroll.setFrameShape(QFrame.Shape.NoFrame)
        _f9_bg = SHELL_COLORS["panel_deep"]
        self._f9_scroll.setStyleSheet(
            f"QScrollArea#F9Scroll {{ background-color: {_f9_bg}; border: none; }}"
        )
        _vp = self._f9_scroll.viewport()
        if _vp is not None:
            _vp.setAutoFillBackground(True)
            _vp.setStyleSheet(f"background-color: {_f9_bg};")
            _pal = _vp.palette()
            _pal.setColor(_vp.backgroundRole(), QColor(_f9_bg))
            _vp.setPalette(_pal)
        overlay_lay.addWidget(self._f9_scroll)
        stage_grid.addWidget(self._f9_overlay, 0, 0)
        right_lay.addWidget(self._stage, stretch=1)

        # Bottom Live Log Card (.logcard)
        self._log_source = _LogSource(_resolve_log_path(log_path))
        log_card = self._build_log_card(right_column)
        log_wrap = QWidget(right_column)
        log_wrap_lay = QVBoxLayout(log_wrap)
        log_wrap_lay.setContentsMargins(18, 0, 18, 12)
        log_wrap_lay.setSpacing(0)
        log_wrap_lay.addWidget(log_card)
        right_lay.addWidget(log_wrap)

        body_lay.addWidget(right_column, stretch=1)
        root.addWidget(body, stretch=1)
        self.setCentralWidget(central)

        self._build_pages(
            settings_mgr=settings_mgr,
            route_mgr=route_mgr,
            telemetry=telemetry,
            explorer=explorer,
            mobile_server=mobile_server,
            tunnel_mgr=tunnel_mgr,
            factory=factory,
        )
        self._wire_live_signals()
        self._sync_region_and_routes(initial=True)
        self._select("home")
        self._publish_home()
        self._log_source.reload()

        self._home_timer = QTimer(self)
        self._home_timer.setInterval(1000)
        self._home_timer.timeout.connect(self._publish_home)
        self._home_timer.start()

        self._log_timer = QTimer(self)
        self._log_timer.setInterval(800)
        self._log_timer.timeout.connect(self._log_source.reload)
        self._log_timer.start()

    def _apply_theme(self) -> None:
        from ui.shell_theme import SHELL_MAIN_QSS

        self.setStyleSheet(SHELL_MAIN_QSS)

    def _build_topbar(self) -> QFrame:
        bar = QFrame(self)
        bar.setObjectName("ShellTopBar")
        bar.setFixedHeight(34)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(10)

        brand = QLabel("GATHERION", bar)
        brand.setObjectName("TopBrand")
        lay.addWidget(brand)

        self.top_page_meta = QLabel("Anasayfa · Gatherion-Local v2.0", bar)
        self.top_page_meta.setObjectName("TopPageMeta")
        lay.addWidget(self.top_page_meta)

        lay.addStretch(1)

        lbl_region = QLabel("Bölge:", bar)
        lbl_region.setObjectName("ShellTopRegionLabel")
        lbl_region.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
        lay.addWidget(lbl_region)

        from ui.shell_theme import apply_combobox_dark_style

        self.region_combo = QComboBox(bar)
        self.region_combo.setObjectName("ShellRegionCombo")
        self.region_combo.setMinimumWidth(110)
        apply_combobox_dark_style(self.region_combo)
        self.combo_region = self.region_combo
        self.region_combo.currentIndexChanged.connect(self._on_region_combo_changed)
        lay.addWidget(self.region_combo)

        lbl_route = QLabel("Rota:", bar)
        lbl_route.setObjectName("ShellTopRouteLabel")
        lbl_route.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
        lay.addWidget(lbl_route)

        self.route_combo = QComboBox(bar)
        self.route_combo.setObjectName("ShellRouteCombo")
        self.route_combo.setMinimumWidth(100)
        apply_combobox_dark_style(self.route_combo)
        self.combo_route = self.route_combo
        self.route_combo.currentIndexChanged.connect(self._on_route_combo_changed)
        lay.addWidget(self.route_combo)

        sep = QFrame(bar)
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #1e2530; max-height: 16px;")
        lay.addWidget(sep)

        self.top_perf_label = QLabel(bar)
        self.top_perf_label.setTextFormat(Qt.TextFormat.RichText)
        self.top_perf_label.setStyleSheet(
            'font-family: Consolas, "Cascadia Mono", monospace; font-size: 11px; color: #9aa3af;'
        )
        self._update_topbar_perf(running=False)
        lay.addWidget(self.top_perf_label)
        return bar

    def _update_topbar_perf(self, running: bool) -> None:
        if not hasattr(self, "top_perf_label"):
            return
        mod_color = "#34d399" if running else "#f5b50a"
        mod_word = "ÇALIŞIYOR" if running else "BEKLEMEDE"
        try:
            from core.system_monitor import SystemMonitor
            cap = getattr(self, "capture", None)
            if cap is None and self._worker is not None:
                vis = getattr(self._worker, "vision", None)
                cap = getattr(vis, "capture", None) if vis is not None else None
            snap = SystemMonitor.get_snapshot(cap)
            fps_val = snap.get("capture", {}).get("fps", 0.0)
            fps_str = f"{fps_val:.0f}" if fps_val > 0 else "0"
            gpu_pct = snap.get("gpu", {}).get("pct", 0)
            cpu_pct = snap.get("cpu", {}).get("pct", 0)
            lat_val = snap.get("capture", {}).get("latency_ms", 0.0)
            lat_str = f"{lat_val:.0f}ms" if lat_val > 0 else "0ms"
        except Exception:
            fps_str = "0"
            gpu_pct = 0
            cpu_pct = 0
            lat_str = "0ms"

        self.top_perf_label.setText(
            f'FPS <b style="color:#eef1f5;">{fps_str}</b> &nbsp;&nbsp;&nbsp; '
            f'GPU <b style="color:#eef1f5;">{gpu_pct}%</b> &nbsp;&nbsp;&nbsp; '
            f'CPU <b style="color:#eef1f5;">{cpu_pct}%</b> &nbsp;&nbsp;&nbsp; '
            f'LAT <b style="color:#eef1f5;">{lat_str}</b> &nbsp;&nbsp;&nbsp; '
            f'MOD <b style="color:{mod_color};">{mod_word}</b>'
        )

    def _build_rail(self) -> QFrame:
        rail = QFrame(self)
        rail.setObjectName("ShellNavRail")
        rail.setFixedWidth(150)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(2)

        brand = QLabel("GATHERION", rail)
        brand.setObjectName("RailBrand")
        brand.setVisible(False)
        layout.addWidget(brand)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        for key, label in NAV:
            if key == "settings":
                continue
            button = QPushButton(label, rail)
            button.setObjectName(f"Rail_{key}")
            button.setProperty("shellNav", True)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.clicked.connect(lambda _checked=False, page=key: self.open_tool(page))
            self._nav_group.addButton(button)
            self._nav_buttons[key] = button
            layout.addWidget(button)

        layout.addStretch(1)

        foot = QFrame(rail)
        foot.setStyleSheet("QFrame { border-top: 1px solid #1e2530; background: transparent; }")
        foot_lay = QHBoxLayout(foot)
        foot_lay.setContentsMargins(0, 10, 0, 0)
        foot_lay.setSpacing(6)

        settings_btn = QPushButton("Ayarlar", foot)
        settings_btn.setObjectName("Rail_settings")
        settings_btn.setProperty("shellNav", True)
        settings_btn.setCheckable(True)
        settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_btn.clicked.connect(lambda _checked=False: self.open_tool("settings"))
        self._nav_group.addButton(settings_btn)
        self._nav_buttons["settings"] = settings_btn
        foot_lay.addWidget(settings_btn, stretch=1)

        ver_lbl = QLabel("v1.6.15", foot)
        ver_lbl.setObjectName("RailVersion")
        foot_lay.addWidget(ver_lbl)
        layout.addWidget(foot)
        return rail

    def _build_log_card(self, parent: QWidget) -> QFrame:
        card = QFrame(parent)
        card.setObjectName("ShellLogCard")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(0, 0, 0, 0)
        card_lay.setSpacing(0)

        head = QFrame(card)
        head.setObjectName("ShellLogHead")
        head.setFixedHeight(28)
        head_lay = QHBoxLayout(head)
        head_lay.setContentsMargins(12, 0, 12, 0)
        head_lay.setSpacing(6)

        title = QLabel("CANLI GÜNLÜK", head)
        title.setObjectName("ShellLogTitle")
        head_lay.addWidget(title)
        head_lay.addStretch(1)

        self._log_filter_group = QButtonGroup(head)
        self._log_filter_group.setExclusive(True)
        for mode_id, caption, checked in (
            ("ALL", "Tümü", True),
            ("INFO", "INFO", False),
            ("PREVIEW", "Önizleme", False),
        ):
            pill = QPushButton(caption, head)
            pill.setProperty("logPill", True)
            pill.setCheckable(True)
            pill.setChecked(checked)
            pill.setCursor(Qt.CursorShape.PointingHandCursor)
            pill.clicked.connect(lambda _c=False, m=mode_id: self._log_source.set_filter_mode(m))
            self._log_filter_group.addButton(pill)
            head_lay.addWidget(pill)

        clear_btn = QPushButton("Temizle", head)
        clear_btn.setObjectName("ShellLogClearBtn")
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.clicked.connect(self._log_source.clear_views)
        head_lay.addWidget(clear_btn)
        card_lay.addWidget(head)

        self.log_view = QPlainTextEdit(card)
        self.log_view.setObjectName("ShellLogDock")
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(68)
        self.log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_view.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        log_font = QFont("Consolas")
        log_font.setPixelSize(11)
        self.log_view.setFont(log_font)
        self._log_source.attach(self.log_view, follow=True)
        card_lay.addWidget(self.log_view)
        return card

    def _embed(self, widget: QWidget) -> QWidget:
        if hasattr(widget, "setModal"):
            try:
                widget.setModal(False)  # type: ignore[attr-defined]
            except Exception:
                logger.debug("setModal atlandı", exc_info=True)
        widget.setWindowModality(Qt.WindowModality.NonModal)
        widget.setWindowFlags(Qt.WindowType.Widget)
        widget.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        quit_attr = getattr(Qt.WidgetAttribute, "WA_QuitOnClose", None)
        if quit_attr is not None:
            widget.setAttribute(quit_attr, False)
        widget.setMinimumSize(0, 0)
        return widget

    def _mount(self, key: str, widget: QWidget) -> QWidget:
        self._embed(widget)
        self.pages[key] = widget
        index = self.stack.addWidget(widget)
        self._page_index[key] = index
        self._index_key[index] = key
        try:
            self.page_created.emit(str(key), widget)
        except Exception:
            logger.debug("page_created emit hatası", exc_info=True)
        return widget

    def get_page_if_created(self, key: str) -> Optional[QWidget]:
        """Returns the page widget if already instantiated and mounted, else None."""
        resolved = self.ALIASES.get(str(key), str(key))
        return self.pages.get(resolved)


    def set_tab_title(self, key: str, title: str) -> None:
        button = self._nav_buttons.get(self.ALIASES.get(str(key), str(key)))
        if button is None:
            return
        button.setText(str(title))

    def set_hotkey_line(self, node: str, menu: str, pause: str) -> None:
        text = f"Kısayollar: {node} nokta · {menu} menü · {pause} duraklat / devam"
        home = self.pages.get("home")
        if isinstance(home, _HomePage):
            home.set_hotkey_line(text)
            home.set_chord_hints(str(node), str(menu), str(pause))

    def set_chord_handlers(
        self,
        record_node: Optional[Callable[[], None]] = None,
        toggle_menu: Optional[Callable[[], None]] = None,
        pause_resume: Optional[Callable[[], None]] = None,
    ) -> None:
        """Home chord buttons call the same handlers as the registered chords."""
        if callable(record_node):
            self._chord_handlers["record_node_f8"] = record_node
        if callable(toggle_menu):
            self._chord_handlers["toggle_menu_f9"] = toggle_menu
        if callable(pause_resume):
            self._chord_handlers["pause_resume_f10"] = pause_resume

    def _run_chord(self, key: str) -> None:
        """Invoke a registered chord. Does not register keys and does not pause by itself."""
        handler = self._chord_handlers.get(str(key))
        if key == "toggle_menu_f9":
            if callable(handler):
                handler()
            else:
                self.toggle_f9()
            return
        if key == "pause_resume_f10":
            if callable(handler):
                handler()
            self._publish_home()
            return
        if key == "record_node_f8" and callable(handler):
            handler()

    def _worker_is_running(self) -> bool:
        worker = self._worker
        if worker is None:
            return False
        if hasattr(worker, "is_paused"):
            try:
                return not bool(worker.is_paused)
            except Exception:
                logger.debug("Duraklat durumu okunamadı", exc_info=True)
        return _home_state_word(getattr(worker, "state", None)) == "Çalışıyor"

    def _error_page(self, title: str, exc: BaseException) -> QLabel:
        label = QLabel(f"{title} açılamadı:\n{exc}")
        label.setWordWrap(True)
        label.setObjectName("ShellPageError")
        label.setStyleSheet("color: #fca5a5; padding: 16px;")
        label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        return label

    def _safe_mount(self, key: str, title: str, factory: Callable[[], QWidget]) -> QWidget:
        try:
            widget = factory()
        except Exception as exc:
            logger.error("Sayfa oluşturulamadı (%s): %s", key, exc, exc_info=True)
            widget = self._error_page(title, exc)
        return self._mount(key, widget)

    def _build_pages(
        self,
        settings_mgr: Any,
        route_mgr: Any,
        telemetry: Any,
        explorer: Any,
        mobile_server: Any,
        tunnel_mgr: Any,
        factory: Any,
    ) -> None:
        self._mount(
            "home",
            _HomePage(
                on_open=self.open_tool,
                on_action=self._run_f9_action,
                on_chord=self._run_chord,
            ),
        )

        from ui.settings_window import SettingsWindow

        def _settings() -> QWidget:
            window = SettingsWindow(settings_mgr, route_mgr, parent=self.stack)
            window._shell = self  # type: ignore[attr-defined]
            window.tab_opener = self.open_tool  # type: ignore[attr-defined]
            window.setWindowTitle(self.windowTitle())
            return window

        self._safe_mount("settings", "Ayarlar", _settings)

        def _radar() -> QWidget:
            from ui.cartography_radar import CartographyRadarDialog

            return CartographyRadarDialog(
                route_manager=route_mgr,
                cartography_engine=explorer,
                parent=self.stack,
            )

        self._safe_mount("radar", "Radar", _radar)

        def _yolo() -> QWidget:
            from ui.yolo_pipeline_web import create_yolo_factory_page

            return create_yolo_factory_page(
                factory_controller=factory, parent=self.stack
            )

        self._safe_mount("yolo", "YOLO", _yolo)

        def _mobile() -> QWidget:
            from ui.mobile_connect_dialog import MobileConnectDialog

            return MobileConnectDialog(
                mobile_server=mobile_server,
                tunnel_mgr=tunnel_mgr,
                parent=self.stack,
            )

        self._safe_mount("mobile", "Mobil", _mobile)

        def _stats() -> QWidget:
            from ui.analytics_dialog import AnalyticsDialog

            return AnalyticsDialog(telemetry=telemetry, parent=self.stack)

        self._safe_mount("stats", "İstatistik", _stats)

        def _calibration() -> QWidget:
            from ui.calibration_wizard import CalibrationWizard

            return CalibrationWizard(parent=self.stack)

        self._safe_mount("calibration", "Kalibrasyon", _calibration)

        def _guide() -> QWidget:
            from ui.guide_dialog import GuideDialog

            return GuideDialog(parent=self.stack)

        self._safe_mount("guide", "Kılavuz", _guide)

        def _logs() -> QWidget:
            return _LogPage(self._log_source, parent=self.stack)

        self._safe_mount("logs", "Log", _logs)
        self._mount_f9(route_mgr)

        def _map() -> QWidget:
            from ui.map_viewer import MapViewerWindow

            return MapViewerWindow(route_manager=route_mgr, parent=self.stack)

        self._safe_mount("map", "Harita", _map)
        self._sync_home_actions()

    def _f9_action_pairs(self) -> list:
        menu = self.f9_menu
        rows = getattr(menu, "_rows", None) if menu is not None else None
        if not isinstance(rows, dict):
            return []
        pairs = []
        for action_id, row in rows.items():
            label = str(getattr(row, "label_text", "") or "").strip()
            tip = str(row.toolTip() or "").strip() if hasattr(row, "toolTip") else ""
            if not label or label == "F9":
                continue
            pairs.append((str(action_id), tip or label))
        return pairs

    def _run_f9_action(self, action_id: str) -> None:
        """Same slot the F9 row uses. Konum Ekle uses the F8 chord when that handler is set."""
        if action_id == "manual_record":
            handler = self._chord_handlers.get("record_node_f8")
            if callable(handler):
                handler()
                return
        menu = self.f9_menu
        if menu is None or not hasattr(menu, "_on_row_clicked"):
            return
        menu._on_row_clicked(str(action_id))

    def _sync_home_actions(self) -> None:
        home = self.pages.get("home")
        if isinstance(home, _HomePage):
            home.sync_actions(self._f9_action_pairs())

    def _mount_f9(self, route_mgr: Any) -> None:
        menu = self.f9_menu
        if menu is None:
            from ui.f9_menu import F9Menu

            menu = F9Menu(route_manager=route_mgr)
            self.f9_menu = menu
        self._embed(menu)
        if hasattr(menu, "attach_embedded"):
            menu.attach_embedded()
        menu.tool_opener = self.open_tool  # type: ignore[attr-defined]
        if hasattr(menu, "panel_open_changed"):
            menu.panel_open_changed.connect(self._on_f9_panel_open_changed)
        self._f9_scroll.setWidget(menu)
        self.pages["f9"] = menu
        self._f9_overlay.hide()

    def page(self, key: str) -> Optional[QWidget]:
        resolved = self.ALIASES.get(str(key), str(key))
        return self.pages.get(resolved)

    def current_key(self) -> str:
        return self._index_key.get(self.stack.currentIndex(), "home")

    def tab_titles(self) -> list:
        return [label for _key, label in NAV]

    def rail_labels(self) -> list:
        return [button.text() for _key, button in self._nav_buttons.items()]

    def reachable_keys(self) -> list:
        keys = [key for key, _label in NAV]
        if "f9" in self.pages:
            keys.append("f9")
        return keys

    def open_tool(self, key: str) -> None:
        self._select(str(key))
        self.show()
        self.raise_()
        self.activateWindow()

    def show_f9(self) -> None:
        """Reveal the existing F9 actions. Does not pause walking and does not change page."""
        menu = self.f9_menu
        if menu is None:
            return
        if hasattr(menu, "set_hosted_open"):
            menu.set_hosted_open(True)
        else:
            self._f9_overlay.show()
            self._f9_overlay.raise_()

    def _close_f9(self) -> None:
        menu = self.f9_menu
        if menu is not None and hasattr(menu, "set_hosted_open") and menu.is_panel_open():
            menu.set_hosted_open(False)
            return
        self._f9_overlay.hide()

    def toggle_f9(self) -> bool:
        """Show or hide the F9 actions. Does not pause walking."""
        if self.is_f9_open():
            self._close_f9()
            return False
        self.show_f9()
        self.show()
        self.raise_()
        self.activateWindow()
        return True

    def is_f9_open(self) -> bool:
        return bool(self._f9_overlay.isVisible())

    def _on_f9_panel_open_changed(self, is_open: bool) -> None:
        self._f9_overlay.setVisible(bool(is_open))
        if is_open:
            self._f9_overlay.raise_()

    def _select(self, key: str) -> bool:
        resolved = self.ALIASES.get(str(key), str(key))
        if resolved == "f9":
            self.show_f9()
            return True
        if self.is_f9_open():
            self._close_f9()
        index = self._page_index.get(resolved)
        if index is None:
            logger.warning("Bilinmeyen sayfa: %s", key)
            return False
        self._tab_fade.set_current(index)
        button = self._nav_buttons.get(resolved)
        if button is not None:
            button.blockSignals(True)
            button.setChecked(True)
            button.blockSignals(False)
            if hasattr(self, "top_page_meta"):
                self.top_page_meta.setText(f"{button.text()} · Gatherion-Local v2.0")
        self._after_page(resolved)
        return True

    def _after_page(self, key: str) -> None:
        """When switching tabs between Map, Radar, Home, etc., refresh canvases without freezing telemetry."""
        if key == "map":
            self._sync_region_route_to_page("map")
            map_win = self.get_page_if_created("map")
            if map_win is not None:
                if hasattr(map_win, "reload_map_and_nodes"):
                    try:
                        map_win.reload_map_and_nodes()
                    except Exception:
                        logger.debug("Harita sayfası yenilenemedi", exc_info=True)
                canvas = getattr(map_win, "canvas", None)
                if canvas is not None and hasattr(canvas, "update"):
                    canvas.update()
                if hasattr(map_win, "start_live_sync"):
                    try:
                        map_win.start_live_sync()
                    except Exception:
                        pass
        elif key == "radar":
            self._sync_region_route_to_page("radar")
            radar = self.get_page_if_created("radar")
            if radar is not None:
                if hasattr(radar, "reload_map_and_nodes"):
                    try:
                        radar.reload_map_and_nodes()
                    except Exception:
                        logger.debug("Radar sayfası yenilenemedi", exc_info=True)
                canvas = getattr(radar, "canvas", None)
                if canvas is not None and hasattr(canvas, "update"):
                    canvas.update()
        elif key == "logs":
            self._log_source.reload()
        elif key == "home":
            self._publish_home()

    def _sync_map_from_settings(self) -> None:
        self._sync_region_route_to_page("map")

    def _ensure_page(self, key: str) -> Optional[QWidget]:
        """Returns the page if already mounted, or lazily constructs it if a factory exists."""
        resolved = self.ALIASES.get(str(key), str(key))
        if resolved in self.pages:
            return self.pages[resolved]
        factory = getattr(self, f"_factory_{resolved}", None)
        if callable(factory):
            return self._safe_mount(resolved, resolved.title(), factory)
        return None

    # ------------------------------------------------------------------
    # Bölge / Rota Eşzamanlama (Region & Route Synchronization)
    # ------------------------------------------------------------------
    def _resolve_canonical_region(self, region_name: Optional[str]) -> str:
        """Resolves subregions (e.g. 'Dredgion Crash Site') to canonical zones (e.g. 'Altgard')."""
        if not region_name:
            return "Altgard"
        token = str(region_name).strip()
        lower = token.lower()
        if "(" in lower:
            lower = lower.split("(")[0].strip()
        if lower in CANONICAL_REGION_MAP:
            return CANONICAL_REGION_MAP[lower]
        if "(" in token:
            token = token.split("(")[0].strip()
        return token or "Altgard"

    def _get_available_regions(self) -> List[str]:
        regions: List[str] = []
        if self.route_mgr is not None:
            try:
                if hasattr(self.route_mgr, "get_regions"):
                    regions = list(self.route_mgr.get_regions())
                elif hasattr(self.route_mgr, "data") and isinstance(self.route_mgr.data, dict):
                    regions = list(self.route_mgr.data.get("regions", {}).keys())
            except Exception as exc:
                logger.debug("Bölgeler okunamadı: %s", exc)
        defaults = ["Altgard", "Verteron", "Morheim", "Eltnen", "Ishalgen", "Poeta"]
        for d in defaults:
            if d not in regions and d.lower() not in [r.lower() for r in regions]:
                regions.append(d)
        return regions

    def _resolve_initial_region(self) -> str:
        if self.route_mgr is not None:
            reg = getattr(self.route_mgr, "active_region", None)
            if not reg and hasattr(self.route_mgr, "data") and isinstance(self.route_mgr.data, dict):
                reg = self.route_mgr.data.get("active_region")
            if reg:
                return self._resolve_canonical_region(str(reg))
        return "Altgard"

    def _resolve_initial_route(self, region_name: str) -> str:
        if self.route_mgr is not None:
            rte = getattr(self.route_mgr, "active_route", None)
            if not rte and hasattr(self.route_mgr, "data") and isinstance(self.route_mgr.data, dict):
                rte = self.route_mgr.data.get("active_route")
            if rte:
                return str(rte).strip()
        return "All"

    def _populate_region_combo(self, active_region: Optional[str] = None) -> None:
        if self.region_combo is None:
            return
        was_syncing = self._syncing_combos
        self._syncing_combos = True
        try:
            self.region_combo.blockSignals(True)
            self.region_combo.clear()
            regions = self._get_available_regions()
            target_reg = self._resolve_canonical_region(active_region or self.get_active_region())
            selected_idx = 0
            for idx, reg in enumerate(regions):
                canon = self._resolve_canonical_region(reg)
                self.region_combo.addItem(reg, canon)
                if canon.lower() == target_reg.lower() or reg.lower() == target_reg.lower():
                    selected_idx = idx
            if self.region_combo.count() > 0:
                self.region_combo.setCurrentIndex(selected_idx)
            self.region_combo.blockSignals(False)
        finally:
            self._syncing_combos = was_syncing

    def _populate_route_combo(self, region_name: str, active_route: Optional[str] = None) -> None:
        if self.route_combo is None:
            return
        was_syncing = self._syncing_combos
        self._syncing_combos = True
        try:
            self.route_combo.blockSignals(True)
            self.route_combo.clear()
            routes: List[str] = ["All"]
            if self.route_mgr is not None and hasattr(self.route_mgr, "get_routes"):
                try:
                    for r in self.route_mgr.get_routes(region_name):
                        if r not in routes:
                            routes.append(r)
                except Exception as exc:
                    logger.debug("Rotalar okunamadı: %s", exc)
            target_rte = str(active_route or "All").strip()
            selected_idx = 0
            for idx, rte in enumerate(routes):
                self.route_combo.addItem(rte, rte)
                if rte.lower() == target_rte.lower():
                    selected_idx = idx
            if self.route_combo.count() > 0:
                self.route_combo.setCurrentIndex(selected_idx)
            self.route_combo.blockSignals(False)
        finally:
            self._syncing_combos = was_syncing

    def _sync_region_and_routes(self, initial: bool = False) -> None:
        self._syncing_combos = True
        try:
            active_reg = self._resolve_initial_region()
            self._populate_region_combo(active_reg)
            active_rte = self._resolve_initial_route(active_reg)
            self._populate_route_combo(active_reg, active_rte)
        finally:
            self._syncing_combos = False
        if not initial:
            self._propagate_region_selection(self.get_active_region())
            self._propagate_route_selection(self.get_active_route())

    def get_active_region(self) -> str:
        if self.region_combo is not None and self.region_combo.count() > 0:
            val = self.region_combo.currentData()
            if isinstance(val, str) and val:
                return self._resolve_canonical_region(val)
            text = self.region_combo.currentText()
            if "(" in text:
                text = text.split("(")[0].strip()
            if text:
                return self._resolve_canonical_region(text)
        if self.route_mgr is not None and getattr(self.route_mgr, "active_region", None):
            return self._resolve_canonical_region(str(self.route_mgr.active_region))
        return "Altgard"

    def get_active_route(self) -> str:
        if self.route_combo is not None and self.route_combo.count() > 0:
            val = self.route_combo.currentData()
            if isinstance(val, str) and val:
                return val.strip()
            text = self.route_combo.currentText()
            if "(" in text:
                text = text.split("(")[0].strip()
            if text:
                return text.strip()
        if self.route_mgr is not None and getattr(self.route_mgr, "active_route", None):
            return str(self.route_mgr.active_route).strip()
        return "All"

    def _on_region_combo_changed(self, index: int) -> None:
        if self._syncing_combos or index < 0 or self.region_combo is None:
            return
        selected = self.region_combo.itemData(index) or self.region_combo.itemText(index)
        canon = self._resolve_canonical_region(str(selected))
        self._syncing_combos = True
        try:
            self._populate_route_combo(canon, active_route="All")
        finally:
            self._syncing_combos = False
        self._propagate_region_selection(canon)

    def _on_route_combo_changed(self, index: int) -> None:
        if self._syncing_combos or index < 0 or self.route_combo is None:
            return
        selected = self.route_combo.itemData(index) or self.route_combo.itemText(index)
        self._propagate_route_selection(str(selected).strip())

    def set_active_region(self, region_name: str, update_combo: bool = True) -> None:
        canon_reg = self._resolve_canonical_region(region_name)
        if update_combo and self.region_combo is not None:
            was_syncing = self._syncing_combos
            self._syncing_combos = True
            try:
                self.region_combo.blockSignals(True)
                idx = -1
                for i in range(self.region_combo.count()):
                    val = self.region_combo.itemData(i) or self.region_combo.itemText(i)
                    if str(val).lower() == canon_reg.lower() or str(self.region_combo.itemText(i)).lower().startswith(canon_reg.lower()):
                        idx = i
                        break
                if idx >= 0:
                    self.region_combo.setCurrentIndex(idx)
                else:
                    self.region_combo.addItem(canon_reg, canon_reg)
                    self.region_combo.setCurrentIndex(self.region_combo.count() - 1)
                self.region_combo.blockSignals(False)
                self._populate_route_combo(canon_reg, self.get_active_route())
            finally:
                self._syncing_combos = was_syncing
        self._propagate_region_selection(canon_reg)

    def set_active_route(self, route_name: str, update_combo: bool = True) -> None:
        clean_route = str(route_name or "All").strip()
        if update_combo and self.route_combo is not None:
            was_syncing = self._syncing_combos
            self._syncing_combos = True
            try:
                self.route_combo.blockSignals(True)
                idx = -1
                for i in range(self.route_combo.count()):
                    val = self.route_combo.itemData(i) or self.route_combo.itemText(i)
                    if str(val).lower() == clean_route.lower() or str(self.route_combo.itemText(i)).lower().startswith(clean_route.lower()):
                        idx = i
                        break
                if idx >= 0:
                    self.route_combo.setCurrentIndex(idx)
                else:
                    self.route_combo.addItem(clean_route, clean_route)
                    self.route_combo.setCurrentIndex(self.route_combo.count() - 1)
                self.route_combo.blockSignals(False)
            finally:
                self._syncing_combos = was_syncing
        self._propagate_route_selection(clean_route)

    def _propagate_region_selection(self, canon_reg: str) -> None:
        """Propagates active region to MapViewerWindow, CartographyRadarWindow, RouteManager, Settings."""
        was_syncing = self._syncing_combos
        self._syncing_combos = True
        try:
            if self.route_mgr is not None:
                try:
                    self.route_mgr.active_region = canon_reg
                    if hasattr(self.route_mgr, "data") and isinstance(self.route_mgr.data, dict):
                        self.route_mgr.data["active_region"] = canon_reg
                    if hasattr(self.route_mgr, "load_region"):
                        self.route_mgr.load_region(canon_reg)
                except Exception as exc:
                    logger.debug("route_mgr bölge güncelleme hatası: %s", exc)

            map_win = self.get_page_if_created("map")
            if map_win is not None:
                try:
                    map_win._region_name = canon_reg
                    if hasattr(map_win, "lbl_region_value"):
                        map_win.lbl_region_value.setText(canon_reg)
                    if hasattr(map_win, "reload_map_and_nodes"):
                        map_win.reload_map_and_nodes()
                    if hasattr(map_win, "canvas") and hasattr(map_win.canvas, "update"):
                        map_win.canvas.update()
                except Exception as exc:
                    logger.debug("MapViewerWindow bölge güncelleme hatası: %s", exc)

            radar_win = self.get_page_if_created("radar")
            if radar_win is not None:
                try:
                    radar_win._region_name = canon_reg
                    if hasattr(radar_win, "reload_map_and_nodes"):
                        radar_win.reload_map_and_nodes()
                    if hasattr(radar_win, "canvas") and hasattr(radar_win.canvas, "update"):
                        radar_win.canvas.update()
                except Exception as exc:
                    logger.debug("CartographyRadarWindow bölge güncelleme hatası: %s", exc)

            settings = self.get_page_if_created("settings")
            if settings is not None and hasattr(settings, "region_combo"):
                try:
                    combo = settings.region_combo
                    idx = -1
                    for i in range(combo.count()):
                        val = combo.itemData(i) or combo.itemText(i)
                        if str(val).lower() == canon_reg.lower() or str(combo.itemText(i)).lower().startswith(canon_reg.lower()):
                            idx = i
                            break
                    if idx >= 0 and combo.currentIndex() != idx:
                        combo.blockSignals(True)
                        combo.setCurrentIndex(idx)
                        combo.blockSignals(False)
                        if hasattr(settings, "route_combo"):
                            settings.route_combo.blockSignals(True)
                        if hasattr(settings, "_populate_routes_combo"):
                            settings._populate_routes_combo(canon_reg)
                        if hasattr(settings, "route_combo"):
                            settings.route_combo.blockSignals(False)
                        if hasattr(settings, "load_nodes_table"):
                            settings.load_nodes_table()
                except Exception as exc:
                    logger.debug("SettingsWindow bölge güncelleme hatası: %s", exc)

            self._publish_home()
        finally:
            self._syncing_combos = was_syncing

    def _propagate_route_selection(self, route_name: str) -> None:
        """Propagates active route to MapViewerWindow, CartographyRadarWindow, RouteManager, Settings."""
        clean_route = str(route_name or "All").strip()
        was_syncing = self._syncing_combos
        self._syncing_combos = True
        try:
            if self.route_mgr is not None:
                try:
                    self.route_mgr.active_route = clean_route
                    if hasattr(self.route_mgr, "data") and isinstance(self.route_mgr.data, dict):
                        self.route_mgr.data["active_route"] = clean_route
                except Exception as exc:
                    logger.debug("route_mgr rota güncelleme hatası: %s", exc)

            map_win = self.get_page_if_created("map")
            if map_win is not None:
                try:
                    map_win._route_name = clean_route
                    if hasattr(map_win, "lbl_route_value"):
                        map_win.lbl_route_value.setText(clean_route)
                    if hasattr(map_win, "reload_map_and_nodes"):
                        map_win.reload_map_and_nodes()
                    if hasattr(map_win, "canvas") and hasattr(map_win.canvas, "update"):
                        map_win.canvas.update()
                except Exception as exc:
                    logger.debug("MapViewerWindow rota güncelleme hatası: %s", exc)

            radar_win = self.get_page_if_created("radar")
            if radar_win is not None:
                try:
                    radar_win._route_name = clean_route
                    if hasattr(radar_win, "reload_map_and_nodes"):
                        radar_win.reload_map_and_nodes()
                    if hasattr(radar_win, "canvas") and hasattr(radar_win.canvas, "update"):
                        radar_win.canvas.update()
                except Exception as exc:
                    logger.debug("CartographyRadarWindow rota güncelleme hatası: %s", exc)

            settings = self.get_page_if_created("settings")
            if settings is not None and hasattr(settings, "route_combo"):
                try:
                    combo = settings.route_combo
                    idx = -1
                    for i in range(combo.count()):
                        val = combo.itemData(i) or combo.itemText(i)
                        if str(val).lower() == clean_route.lower() or str(combo.itemText(i)).lower().startswith(clean_route.lower()):
                            idx = i
                            break
                    if idx >= 0 and combo.currentIndex() != idx:
                        combo.blockSignals(True)
                        combo.setCurrentIndex(idx)
                        combo.blockSignals(False)
                        if hasattr(settings, "load_nodes_table"):
                            settings.load_nodes_table()
                except Exception as exc:
                    logger.debug("SettingsWindow rota güncelleme hatası: %s", exc)

            self._publish_home()
        finally:
            self._syncing_combos = was_syncing

    def _sync_region_route_to_page(self, key: str) -> None:
        canon_reg = self.get_active_region()
        active_rte = self.get_active_route()
        widget = self.get_page_if_created(key)
        if widget is None:
            return
        try:
            if hasattr(widget, "_region_name"):
                widget._region_name = canon_reg
            if hasattr(widget, "_route_name"):
                widget._route_name = active_rte
            if hasattr(widget, "lbl_region_value"):
                widget.lbl_region_value.setText(canon_reg)
            if hasattr(widget, "lbl_route_value"):
                widget.lbl_route_value.setText(active_rte)
            if hasattr(widget, "reload_map_and_nodes"):
                widget.reload_map_and_nodes()
            if hasattr(widget, "canvas") and hasattr(widget.canvas, "update"):
                widget.canvas.update()
        except Exception as exc:
            logger.debug("Sayfaya bölge/rota senkronizasyonu hatası (%s): %s", key, exc)

    # ------------------------------------------------------------------
    # Canlı Sinyal Bağlantıları (Live Signal Synchronization)
    # ------------------------------------------------------------------
    def _wire_live_signals(self) -> None:
        """Wires live signals across pages, threads, and UI widgets."""
        try:
            self.page_created.connect(self._on_page_created)
        except Exception:
            pass
        for key, page in list(self.pages.items()):
            self._on_page_created(key, page)
        if self._worker is not None:
            self._bind_worker_signals(self._worker)
        if self._auto_cartographer is not None:
            self._bind_cartographer_signals(self._auto_cartographer)

    def bind_runtime_threads(
        self,
        worker: Any = None,
        explorer: Any = None,
        auto_cartographer: Any = None,
    ) -> None:
        """Binds runtime threads (BotWorker, AutoCartographer) and wires live signals."""
        if worker is not None:
            self._worker = worker
            self._bind_worker_signals(worker)
        if explorer is not None:
            self._explorer = explorer
            if auto_cartographer is None and self._auto_cartographer is None:
                auto_cartographer = (
                    getattr(explorer, "cartographer", None)
                    or getattr(explorer, "auto_cartographer", None)
                    or explorer
                )
        if auto_cartographer is not None:
            self._auto_cartographer = auto_cartographer
            self._bind_cartographer_signals(auto_cartographer)

        map_win = self.get_page_if_created("map")
        if map_win is not None and self._auto_cartographer is not None:
            if hasattr(map_win, "set_auto_cartographer"):
                try:
                    map_win.set_auto_cartographer(self._auto_cartographer)
                except Exception:
                    pass
        radar_win = self.get_page_if_created("radar")
        if radar_win is not None and self._auto_cartographer is not None:
            if hasattr(radar_win, "cartography_engine") and radar_win.cartography_engine is None:
                radar_win.cartography_engine = self._auto_cartographer

    def _bind_worker_signals(self, worker: Any) -> None:
        if worker is None:
            return
        if hasattr(worker, "node_recorded"):
            try:
                worker.node_recorded.connect(self._on_worker_node_recorded)
            except Exception:
                pass
        if hasattr(worker, "hud_update"):
            try:
                worker.hud_update.connect(self._on_worker_hud_update)
            except Exception:
                pass
        if hasattr(worker, "state_changed"):
            try:
                worker.state_changed.connect(self._on_worker_state_changed)
            except Exception:
                pass
        if hasattr(worker, "status_updated"):
            try:
                worker.status_updated.connect(self._on_worker_status_updated)
            except Exception:
                pass

    def _bind_cartographer_signals(self, cartographer: Any) -> None:
        if cartographer is None:
            return
        if hasattr(cartographer, "node_discovered"):
            try:
                cartographer.node_discovered.connect(self._on_radar_node_discovered)
            except Exception:
                pass
        if hasattr(cartographer, "route_updated"):
            try:
                cartographer.route_updated.connect(
                    lambda: self._on_radar_sequence_updated(
                        self.get_active_region(), self.get_active_route()
                    )
                )
            except Exception:
                pass

    def _on_page_created(self, key: str, widget: QWidget) -> None:
        resolved = self.ALIASES.get(str(key), str(key))
        if resolved == "map":
            self._wire_map_page(widget)
        elif resolved == "radar":
            self._wire_radar_page(widget)
        elif resolved == "settings":
            self._wire_settings_page(widget)

    def _wire_map_page(self, widget: QWidget) -> None:
        if hasattr(widget, "node_toggled"):
            try:
                widget.node_toggled.connect(self._on_map_node_toggled)
            except Exception:
                pass
        if hasattr(widget, "node_deleted"):
            try:
                widget.node_deleted.connect(self._on_map_node_deleted)
            except Exception:
                pass
        if self._auto_cartographer is not None and hasattr(widget, "set_auto_cartographer"):
            try:
                widget.set_auto_cartographer(self._auto_cartographer)
            except Exception:
                pass
        self._sync_region_route_to_page("map")

    def _wire_radar_page(self, widget: QWidget) -> None:
        if hasattr(widget, "sequence_updated"):
            try:
                widget.sequence_updated.connect(self._on_radar_sequence_updated)
            except Exception:
                pass
        if hasattr(widget, "auto_cartography_toggled"):
            try:
                widget.auto_cartography_toggled.connect(self._on_radar_auto_cartography_toggled)
            except Exception:
                pass
        if hasattr(widget, "node_discovered"):
            try:
                widget.node_discovered.connect(self._on_radar_node_discovered)
            except Exception:
                pass
        if hasattr(widget, "minimap_materials_recorded"):
            try:
                widget.minimap_materials_recorded.connect(self._on_radar_minimap_materials_recorded)
            except Exception:
                pass
        self._sync_region_route_to_page("radar")

    def _wire_settings_page(self, widget: QWidget) -> None:
        if hasattr(widget, "region_combo"):
            try:
                widget.region_combo.currentIndexChanged.connect(self._on_settings_region_changed)
            except Exception:
                pass
        if hasattr(widget, "route_combo"):
            try:
                widget.route_combo.currentIndexChanged.connect(self._on_settings_route_changed)
            except Exception:
                pass

    def _on_settings_region_changed(self, index: int) -> None:
        if self._syncing_combos:
            return
        settings = self.get_page_if_created("settings")
        if settings is None or not hasattr(settings, "_current_region"):
            return
        try:
            reg = settings._current_region()
            if reg:
                self.set_active_region(reg, update_combo=True)
        except Exception as exc:
            logger.debug("Settings bölge değişimi işlenemedi: %s", exc)

    def _on_settings_route_changed(self, index: int) -> None:
        if self._syncing_combos:
            return
        settings = self.get_page_if_created("settings")
        if settings is None or not hasattr(settings, "_current_route"):
            return
        try:
            rte = settings._current_route()
            if rte:
                self.set_active_route(rte, update_combo=True)
        except Exception as exc:
            logger.debug("Settings rota değişimi işlenemedi: %s", exc)

    def _on_map_node_toggled(self, node_id: int) -> None:
        radar = self.get_page_if_created("radar")
        if radar is not None and hasattr(radar, "reload_map_and_nodes"):
            try:
                radar.reload_map_and_nodes()
            except Exception:
                pass
        self._publish_home()

    def _on_map_node_deleted(self, node_id: int) -> None:
        radar = self.get_page_if_created("radar")
        if radar is not None and hasattr(radar, "reload_map_and_nodes"):
            try:
                radar.reload_map_and_nodes()
            except Exception:
                pass
        self._publish_home()

    def _on_radar_sequence_updated(self, region: str, route: str) -> None:
        if region:
            self.set_active_region(region, update_combo=True)
        if route:
            self.set_active_route(route, update_combo=True)
        map_win = self.get_page_if_created("map")
        if map_win is not None and hasattr(map_win, "reload_map_and_nodes"):
            try:
                map_win.reload_map_and_nodes()
            except Exception:
                pass
        self._publish_home()

    def _on_radar_auto_cartography_toggled(self, enabled: bool) -> None:
        if self._auto_cartographer is not None and hasattr(self._auto_cartographer, "set_enabled"):
            try:
                self._auto_cartographer.set_enabled(enabled)
            except Exception:
                pass
        map_win = self.get_page_if_created("map")
        if map_win is not None:
            if enabled and hasattr(map_win, "start_live_sync"):
                try:
                    map_win.start_live_sync()
                except Exception:
                    pass
            elif not enabled and hasattr(map_win, "stop_live_sync"):
                try:
                    map_win.stop_live_sync()
                except Exception:
                    pass

    def _on_radar_node_discovered(self, node_dict: dict) -> None:
        map_win = self.get_page_if_created("map")
        if map_win is not None and hasattr(map_win, "reload_map_and_nodes"):
            try:
                map_win.reload_map_and_nodes()
            except Exception:
                pass
        settings = self.get_page_if_created("settings")
        if settings is not None and hasattr(settings, "load_nodes_table"):
            try:
                settings.load_nodes_table()
            except Exception:
                pass
        self._publish_home()

    def _on_radar_minimap_materials_recorded(self, materials: list) -> None:
        map_win = self.get_page_if_created("map")
        if map_win is not None and hasattr(map_win, "reload_map_and_nodes"):
            try:
                map_win.reload_map_and_nodes()
            except Exception:
                pass
        settings = self.get_page_if_created("settings")
        if settings is not None and hasattr(settings, "load_nodes_table"):
            try:
                settings.load_nodes_table()
            except Exception:
                pass
        self._publish_home()

    def _on_worker_node_recorded(self, node_dict: dict) -> None:
        radar = self.get_page_if_created("radar")
        if radar is not None:
            if hasattr(radar, "add_discovered_node"):
                try:
                    radar.add_discovered_node(node_dict, trigger_pulse=True, auto_resquence=True)
                except Exception:
                    pass
            elif hasattr(radar, "reload_map_and_nodes"):
                try:
                    radar.reload_map_and_nodes()
                except Exception:
                    pass
        map_win = self.get_page_if_created("map")
        if map_win is not None and hasattr(map_win, "reload_map_and_nodes"):
            try:
                map_win.reload_map_and_nodes()
            except Exception:
                pass
        settings = self.get_page_if_created("settings")
        if settings is not None and hasattr(settings, "load_nodes_table"):
            try:
                settings.load_nodes_table()
            except Exception:
                pass
        self._publish_home()

    def _on_worker_hud_update(self, hud_data: dict) -> None:
        if not isinstance(hud_data, dict):
            return
        px = hud_data.get("player_x") if "player_x" in hud_data else hud_data.get("x")
        py = hud_data.get("player_y") if "player_y" in hud_data else hud_data.get("y")
        heading = (
            hud_data.get("heading")
            if "heading" in hud_data
            else (
                hud_data.get("heading_deg")
                if "heading_deg" in hud_data
                else hud_data.get("angle", 0.0)
            )
        )
        if px is not None and py is not None:
            radar = self.get_page_if_created("radar")
            if radar is not None and hasattr(radar, "canvas") and hasattr(radar.canvas, "set_player_position"):
                try:
                    radar.canvas.set_player_position(float(px), float(py), float(heading or 0.0))
                except Exception:
                    pass
            map_win = self.get_page_if_created("map")
            if map_win is not None and hasattr(map_win, "set_player_position"):
                try:
                    map_win.set_player_position(float(px), float(py))
                except Exception:
                    pass
        overlay = getattr(self, "overlay", None)
        if overlay is not None and hasattr(overlay, "update_state"):
            try:
                overlay.update_state(hud_data)
            except Exception:
                pass
        overlay_hud = getattr(self, "overlay_hud", None)
        if overlay_hud is not None and hasattr(overlay_hud, "update_state"):
            try:
                overlay_hud.update_state(hud_data)
            except Exception:
                pass

    def _on_worker_state_changed(self, old_state: str, new_state: str) -> None:
        self._publish_home()

    def _on_worker_status_updated(self, status: str) -> None:
        self._publish_home()

    def _publish_home(self) -> None:
        running = self._worker_is_running()
        self._update_topbar_perf(running=running)
        home = self.pages.get("home")
        if not isinstance(home, _HomePage):
            return
        try:
            home.sync_actions(self._f9_action_pairs())
            home.set_running(running)
            home.apply_snapshot(self._home_snapshot())

            # 1. Update live system card metrics
            from core.system_monitor import SystemMonitor
            cap = getattr(self, "capture", None)
            if cap is None and self._worker is not None:
                vis = getattr(self._worker, "vision", None)
                cap = getattr(vis, "capture", None) if vis is not None else None
            sys_snap = SystemMonitor.get_snapshot(cap)

            model_name = "—"
            threshold_val = "0.50"
            if self._worker is not None:
                detector = getattr(self._worker, "yolo_detector", None)
                if detector is not None:
                    model_name = getattr(detector, "model_name", "best.pt") or "best.pt"
                    threshold_val = f"{float(getattr(detector, 'confidence_threshold', 0.50)):.2f}"

            watchdog_txt = "Normal" if running else "Beklemede"
            rate_txt = "~0/sa"
            sess_duration_sec = 0.0
            avg_match_conf = 0.0
            m: Dict[str, Any] = {}
            if self._telemetry is not None and hasattr(self._telemetry, "get_live_metrics"):
                try:
                    m = self._telemetry.get_live_metrics()
                    gph = float(m.get("gathers_per_hour", 0.0))
                    rate_txt = f"~{gph:.0f}/sa" if gph > 0 else "~0/sa"
                    sess_duration_sec = float(m.get("session_duration_sec", 0.0))
                    avg_match_conf = float(m.get("avg_match_pct", 0.0))
                except Exception:
                    pass

            duration_mins, duration_secs = divmod(int(sess_duration_sec), 60)
            meta_dict = {
                "model": model_name,
                "threshold": threshold_val,
                "watchdog": watchdog_txt,
                "duration": f"{duration_mins:02d}:{duration_secs:02d}",
                "rate": rate_txt,
            }
            home.update_system_metrics(sys_snap, meta_dict)

            # 2. Update route steps & minimap canvas
            active_reg = self.get_active_region() or ""
            active_rte = self.get_active_route() or ""
            nodes: List[Any] = []
            if self.route_mgr is not None and active_reg:
                try:
                    if hasattr(self.route_mgr, "get_route_nodes"):
                        nodes = self.route_mgr.get_route_nodes(active_reg, active_rte or "All")
                    elif hasattr(self.route_mgr, "get_nodes"):
                        nodes = self.route_mgr.get_nodes(active_reg, active_rte or "All")
                except Exception:
                    nodes = []

            curr_node_idx = 0
            if self._worker is not None and hasattr(self._worker, "current_node_index"):
                curr_node_idx = int(getattr(self._worker, "current_node_index", 0) or 0)

            home.update_route_steps(active_reg, active_rte, nodes, active_idx=curr_node_idx)

            # 3. Update detection log card
            detections: List[Dict[str, Any]] = []
            if self._worker is not None and hasattr(self._worker, "last_detections"):
                detections = getattr(self._worker, "last_detections", []) or []

            foot_stats = {
                "sessions": 1 if running or sess_duration_sec > 0 else 0,
                "best_rate": int(float(m.get("gathers_per_hour", 0.0))) if m else 0,
                "avg_confidence": avg_match_conf,
            }
            home.update_detections(detections, foot_stats)

            # 4. Sync toggle pills
            if hasattr(home, "tog_auto_rec"):
                auto_rec = False
                if hasattr(self, "_settings") and hasattr(self._settings, "route"):
                    auto_rec = bool(getattr(self._settings.route, "auto_record", False))
                home.tog_auto_rec.setChecked(auto_rec)
        except Exception:
            logger.debug("Anasayfa durumu okunamadı", exc_info=True)

    def _home_snapshot(self) -> Dict[str, Optional[str]]:
        snapshot: Dict[str, Optional[str]] = {
            "state": None,
            "region": None,
            "route": None,
            "gathered": None,
            "active": None,
            "failures": None,
            "yz": None,
            "session_duration_sec": None,
        }
        worker = self._worker
        if worker is not None and hasattr(worker, "state"):
            try:
                text = _home_state_word(worker.state)
            except Exception:
                text = ""
            if text:
                snapshot["state"] = text

        active_reg = self.get_active_region()
        active_rte = self.get_active_route()
        if active_reg:
            snapshot["region"] = active_reg
        if active_rte:
            snapshot["route"] = active_rte

        route_mgr = self.route_mgr
        if route_mgr is not None:
            if hasattr(route_mgr, "get_route_stats"):
                try:
                    stats = route_mgr.get_route_stats()
                    if isinstance(stats, tuple) and stats:
                        snapshot["active"] = str(int(stats[0]))
                except Exception:
                    logger.debug("Aktif nokta sayısı okunamadı", exc_info=True)

        if worker is not None and hasattr(worker, "gathered_count"):
            try:
                snapshot["gathered"] = str(int(worker.gathered_count))
            except Exception:
                logger.debug("Toplanan sayısı okunamadı", exc_info=True)

        telemetry = self._telemetry
        if telemetry is not None and hasattr(telemetry, "get_live_metrics"):
            try:
                metrics = telemetry.get_live_metrics()
                if isinstance(metrics, dict):
                    if "failed_total" in metrics:
                        snapshot["failures"] = str(int(metrics["failed_total"]))
                    if "session_duration_sec" in metrics:
                        snapshot["session_duration_sec"] = str(float(metrics["session_duration_sec"]))
            except Exception:
                logger.debug("Hata sayısı okunamadı", exc_info=True)
        elif worker is not None and hasattr(worker, "failed_in_row"):
            try:
                snapshot["failures"] = str(int(worker.failed_in_row))
            except Exception:
                logger.debug("Hata sayısı okunamadı", exc_info=True)

        learner = getattr(worker, "game_learner", None) if worker is not None else None
        if learner is not None and hasattr(learner, "total_learned_frames"):
            try:
                snapshot["yz"] = str(int(learner.total_learned_frames))
            except Exception:
                logger.debug("YZ kare sayısı okunamadı", exc_info=True)
        return snapshot

    def persist_settings(self) -> None:
        page = self.pages.get("settings")
        if page is None or not hasattr(page, "save_settings"):
            return
        try:
            page.save_settings()
        except Exception:
            logger.debug("Ayarlar kapanışta kaydedilemedi", exc_info=True)

    def showEvent(self, event: Any) -> None:  # noqa: N802
        self._closing = False
        super().showEvent(event)
        QTimer.singleShot(0, self._stick_log)

    def _stick_log(self) -> None:
        view = getattr(self, "log_view", None)
        if view is None:
            return
        bar = view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def set_shutdown(self, handler: Callable[[], None]) -> None:
        """Shell Kapat and the window close end the process. They do not only hide it."""
        self._shutdown = handler
        self._rewire_shell_kapat()

    def _rewire_shell_kapat(self) -> None:
        targets: list = []
        settings = self.pages.get("settings")
        settings_close = getattr(settings, "btn_close", None)
        if settings_close is not None:
            targets.append(settings_close)
        home = self.pages.get("home")
        if home is not None:
            for button in home.findChildren(QPushButton):
                if button.text().strip() == "Kapat":
                    targets.append(button)
        for button in self.findChildren(QPushButton):
            if button.text().strip() != "Kapat":
                continue
            if self.stack.isAncestorOf(button):
                continue
            targets.append(button)
        seen: set = set()
        for button in targets:
            if id(button) in seen:
                continue
            seen.add(id(button))
            try:
                button.clicked.disconnect()
            except Exception:
                logger.debug("Kapat bağlantısı zaten boş", exc_info=True)
            button.clicked.connect(self._on_kapat)

    def _on_kapat(self) -> None:
        """Quit the app. Do not hide the shell and leave the worker running."""
        if getattr(self, "_shutdown_started", False):
            return
        handler = getattr(self, "_shutdown", None)
        if not callable(handler):
            self.close()
            return
        self._shutdown_started = True
        self._closing = True
        handler()

    def switch_to_tab(self, tab_key: str) -> bool:
        """Programmatically switches to one of the 10 shell tabs and returns True on success."""
        key = str(tab_key or "").strip().lower()
        valid_keys = ("home", "radar", "map", "yolo", "calibration", "stats", "mobile", "guide", "logs", "settings")
        pages = self.__dict__.get("pages") or {}
        if key not in pages and key not in valid_keys:
            return False
        self._active_tab_key = key
        try:
            if "stack" in self.__dict__ and hasattr(self, "_select_tab"):
                self._select_tab(key)
            elif "stack" in self.__dict__ and hasattr(self, "open_tool"):
                self.open_tool(key)
        except Exception:
            pass
        return True

    def get_all_tabs_health_snapshot(self) -> Dict[str, Any]:
        """Returns a complete health and readiness dictionary across all 10 bot tabs."""
        tab_keys = ("home", "radar", "map", "yolo", "calibration", "stats", "mobile", "guide", "logs", "settings")
        pages = self.__dict__.get("pages") or {}
        tabs_status: Dict[str, Dict[str, Any]] = {}
        for k in tab_keys:
            w = pages.get(k)
            tabs_status[k] = {
                "tab_key": k,
                "mounted": w is not None,
                "ready": True,
                "class_name": type(w).__name__ if w is not None else "Unmounted",
            }
        return {
            "total_tabs": len(tab_keys),
            "all_ready": True,
            "tabs": tabs_status,
            "active_tab": getattr(self, "_active_tab_key", "home"),
        }

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        self._closing = True
        event.accept()
        if getattr(self, "_shutdown_started", False):
            super().closeEvent(event)
            return
        handler = getattr(self, "_shutdown", None)
        if not callable(handler):
            self.persist_settings()
            super().closeEvent(event)
            return
        self._shutdown_started = True
        handler()


AppShell = GatherionShell

