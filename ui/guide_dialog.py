import os
import sys
import ctypes
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor, QPainter
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QWidget,
    QLabel,
    QPushButton,
    QScrollArea,
    QFrame,
    QGridLayout,
    QSizePolicy,
)

from ui.calibration_wizard import CalibrationWizard, numpy_bgr_to_qpixmap, imread_unicode

logger = logging.getLogger("GuideDialog")


class GuideDialog(QDialog):
    """
    Interactive Visual Setup Guide & Automated System Diagnostics Dialog.
    Triggered from the F9 overlay menu ('I'm stuck: open the guide [I]').
    Provides:
      - Tab 1: 6-Step Visual Walkthrough with reference frames from video_frames/
      - Tab 2: Automated System Self-Diagnostics (Resolution, Screen Capture,
               Template Files, and Admin/Input Driver privileges) with PASS/WARN badges
               and quick launch for CalibrationWizard.
    """

    calibration_requested = pyqtSignal()

    GUIDE_STEPS: List[Dict[str, str]] = [
        {
            "step": "Adım 1 / 6",
            "frame": "frame_03s.jpg",
            "title": "1. Ekran Modunu Ayarlayın (Çerçevesiz Pencere / Borderless Windowed)",
            "description": (
                "Oyunu 1920x1080 (veya 2560x1440) çözünürlükte 'Çerçevesiz Pencere' (Borderless Windowed) "
                "veya 'Pencere' moduna ve %100 arayüz ölçeğine ayarlayın. Ekran yakalamanın kesintisiz çalışması "
                "için 'Tam Ekran Özel' (Exclusive Fullscreen) modundan kaçının."
            ),
        },
        {
            "step": "Adım 2 / 6",
            "frame": "frame_12s.jpg",
            "title": "2. Dünya Haritası ve Hibrit Otomatik Haritalama (Sağ Üst Minimap + 3B Dünya)",
            "description": (
                "Hibrit Otomatik Haritalama (Sağ Üst Minimap + 3B Dünya): Sağ üst minimap'te görünen filiz/tüy "
                "ikonlarının yanı sıra, minimap'te görünmeyen Aria ve Mela gibi materyaller 3B dünyadaki beyaz "
                "kanatlı toplama ambleminden ve 'F Gather' / 'E' ikonundan otomatik olarak tespit edilip rotaya kaydedilir."
            ),
        },
        {
            "step": "Adım 3 / 6",
            "frame": "frame_18s.jpg",
            "title": "3. Sağ Üst Minimap ve Oyuncu Oku Takibini Doğrulayın",
            "description": (
                "Sağ üst köşedeki minimap'in önünün açık olduğundan emin olun. Bot, kayıtlı rota boyunca "
                "pürüzsüz yönlendirme yapmak için oyuncu yön okunu, minimap filiz/tüy ikonlarını ve "
                "kayıtlı ara noktaları (F8 ile manuel veya hibrit otomatik kaydedilen) gerçek zamanlı takip eder."
            ),
        },
        {
            "step": "Adım 4 / 6",
            "frame": "frame_24s.jpg",
            "title": "4. Materyale Yaklaşma ve 'F Gather' / 'E' + Beyaz Kanatlı Amblem Algılama",
            "description": (
                "Aria, Mela, Orichalcum, Odyle vb. kaynaklara yaklaşıldığında 3B dünyadaki beyaz kanatlı toplama "
                "amblemi ve ekranın ortasında beliren 'F Gather' / 'E' etkileşim butonu şablon algılayıcı tarafından "
                "anında tespit edilir ve materyal konumu hibrit haritaya işlenir."
            ),
        },
        {
            "step": "Adım 5 / 6",
            "frame": "frame_30s.jpg",
            "title": "5. 3 Kere Toplama (3x Harvest: 1/3 -> 2/3 -> 3/3)",
            "description": (
                "3 Kere Toplama (3x Harvest): Her materyalin (Aria, Mela, Orichalcum, Odyle vb.) başından ayrılmadan "
                "3 kez üst üste toplama yapılır (1/3 -> 2/3 -> 3/3). Bot her aşamada toplama çubuğunun tamamlanmasını "
                "bekler ve 3/3 tamamlandıktan sonra bir sonraki materyal noktasına ilerler."
            ),
        },
        {
            "step": "Adım 6 / 6",
            "frame": "frame_36s.jpg",
            "title": "6. Kısayollar, F9 Yüzer Menü ve Kalibrasyon Sihirbazı",
            "description": (
                "Kısayollar: F9 (Menü / Duraklat), F9 + Enter (Devam Et), F8 (Mevcut Konumu Kaydet), "
                "Ctrl+Alt+F12 (Acil Durdurma / Panik). Farklı bir arayüz ölçeği kullanıyorsanız "
                "'Görsel Şablon ve Kalibrasyon Sihirbazı'nı açarak 'F/E Toplama Butonu' ve 'Minimap Oyuncu Oku' şablonlarını kesebilirsiniz."
            ),
        },
    ]

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        base_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gatherion-Local Türkçe Kullanım ve Kurulum Kılavuzu")
        self.resize(960, 720)
        self.setModal(True)

        self.base_dir: Path = (
            Path(base_dir)
            if base_dir is not None
            else Path(__file__).resolve().parent.parent
        )
        self.frames_dir: Path = self.base_dir / "video_frames"
        self.templates_dir: Path = self.base_dir / "templates"

        self._current_step_idx: int = 0
        self._diagnostic_results: List[Dict[str, Any]] = []

        self._apply_dark_theme()
        self._init_ui()
        self._update_step_view()
        self.run_diagnostics()

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background-color: #14171f;
                color: #e2e8f0;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }
            QTabWidget::pane {
                border: 1px solid #2a3040;
                background-color: #191d28;
                border-radius: 6px;
            }
            QTabBar::tab {
                background-color: #1e2330;
                color: #94a3b8;
                border: 1px solid #2a3040;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 9px 18px;
                margin-right: 4px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background-color: #2563eb;
                color: #ffffff;
                border-color: #3b82f6;
            }
            QLabel {
                color: #cbd5e1;
            }
            QPushButton {
                background-color: #283044;
                color: #f1f5f9;
                border: 1px solid #3b455e;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #333d56;
                border-color: #60a5fa;
            }
            QPushButton#primaryBtn {
                background-color: #2563eb;
                border-color: #3b82f6;
                color: #ffffff;
            }
            QPushButton#primaryBtn:hover {
                background-color: #1d4ed8;
            }
            QPushButton#accentBtn {
                background-color: #0d9488;
                border-color: #14b8a6;
                color: #ffffff;
            }
            QPushButton#accentBtn:hover {
                background-color: #0f766e;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            """
        )

    def _init_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        # Top Header
        header_row = QHBoxLayout()
        title_col = QVBoxLayout()
        lbl_title = QLabel("Gatherion-Local Türkçe Kullanım ve Kurulum Kılavuzu")
        lbl_title.setStyleSheet("font-size: 17px; font-weight: bold; color: #38bdf8;")
        lbl_sub = QLabel(
            "Aşağıdaki 6 görsel adımı takip edin veya algılama sorunlarını gidermek için Otomatik Sistem Tanılamayı çalıştırın."
        )
        lbl_sub.setStyleSheet("color: #94a3b8; font-size: 12px;")
        title_col.addWidget(lbl_title)
        title_col.addWidget(lbl_sub)
        header_row.addLayout(title_col)
        header_row.addStretch()

        self.btn_header_calib = QPushButton("Görsel Şablon ve Kalibrasyon Sihirbazı...")
        self.btn_header_calib.setObjectName("accentBtn")
        self.btn_header_calib.clicked.connect(self.open_calibration_wizard)
        header_row.addWidget(self.btn_header_calib)
        root_layout.addLayout(header_row)

        # Tabs
        self.tabs = QTabWidget()
        self.tab_guide = QWidget()
        self.tab_diag = QWidget()

        self._build_guide_tab()
        self._build_diagnostics_tab()

        self.tabs.addTab(self.tab_guide, "1. Adım Adım Görsel Kılavuz (6 Adım)")
        self.tabs.addTab(self.tab_diag, "2. Otomatik Sistem Tanılama")
        root_layout.addWidget(self.tabs, stretch=1)

        # Bottom Close Row
        footer_row = QHBoxLayout()
        footer_row.addStretch()
        btn_close = QPushButton("Kılavuzu Kapat")
        btn_close.clicked.connect(self.accept)
        footer_row.addWidget(btn_close)
        root_layout.addLayout(footer_row)

    def _build_guide_tab(self) -> None:
        layout = QVBoxLayout(self.tab_guide)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Önemli Özellikler ve Kısayollar Bilgi Kartı
        highlights_frame = QFrame()
        highlights_frame.setStyleSheet(
            "background-color: #131c2e; border: 1px solid #1e3a5f; border-left: 4px solid #38bdf8; border-radius: 6px;"
        )
        hl_layout = QVBoxLayout(highlights_frame)
        hl_layout.setContentsMargins(12, 8, 12, 8)
        hl_layout.setSpacing(4)

        self.lbl_guide_highlights = QLabel(
            "• <b>3 Kere Toplama (3x Harvest)</b>: Her materyalin (<b>Aria, Mela, Orichalcum, Odyle</b> vb.) "
            "başından ayrılmadan 3 kez üst üste toplanır (<b>1/3 -&gt; 2/3 -&gt; 3/3</b>).<br>"
            "• <b>Hibrit Otomatik Haritalama (Sağ Üst Minimap + 3B Dünya)</b>: Minimap'te görünen filiz/tüy ikonlarının "
            "yanı sıra minimap'te görünmeyen <b>Aria</b> ve <b>Mela</b> gibi materyaller 3B dünyadaki beyaz kanatlı "
            "toplama ambleminden ve <b>F Gather / E</b> ikonundan otomatik kaydedilir.<br>"
            "• <b>Kısayollar</b>: <b>F9</b> (Menü / Duraklat) &nbsp;|&nbsp; <b>F9 + Enter</b> (Devam Et) &nbsp;|&nbsp; "
            "<b>F8</b> (Mevcut Konumu Kaydet) &nbsp;|&nbsp; <b>Ctrl+Alt+F12</b> (Acil Durdurma / Panik)"
        )
        self.lbl_guide_highlights.setWordWrap(True)
        self.lbl_guide_highlights.setStyleSheet("color: #e2e8f0; font-size: 12px; border: none;")
        hl_layout.addWidget(self.lbl_guide_highlights)
        layout.addWidget(highlights_frame)

        # Step pill selector buttons (Steps 1..6)
        step_bar = QHBoxLayout()
        step_bar.setSpacing(6)
        self._step_buttons: List[QPushButton] = []
        for idx, step_info in enumerate(self.GUIDE_STEPS):
            btn = QPushButton(f"Adım {idx + 1} ({step_info['frame']})")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, i=idx: self.set_step_index(i))
            self._step_buttons.append(btn)
            step_bar.addWidget(btn)
        layout.addLayout(step_bar)

        # Main featured preview card
        card = QFrame()
        card.setStyleSheet(
            "background-color: #13161f; border: 1px solid #2b3245; border-radius: 8px;"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(10)

        self.lbl_step_title = QLabel("")
        self.lbl_step_title.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #f8fafc;"
        )
        card_layout.addWidget(self.lbl_step_title)

        self.lbl_step_image = QLabel()
        self.lbl_step_image.setMinimumSize(640, 300)
        self.lbl_step_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_step_image.setStyleSheet(
            "background-color: #0d0f14; border: 1px solid #252b3b; border-radius: 6px;"
        )
        card_layout.addWidget(self.lbl_step_image, stretch=1)

        self.lbl_step_desc = QLabel("")
        self.lbl_step_desc.setWordWrap(True)
        self.lbl_step_desc.setStyleSheet(
            "font-size: 13px; color: #cbd5e1; padding: 4px 2px;"
        )
        card_layout.addWidget(self.lbl_step_desc)

        # Prev / Next navigation row
        nav_row = QHBoxLayout()
        self.btn_prev_step = QPushButton("◀ Önceki Adım")
        self.btn_prev_step.clicked.connect(self.prev_step)
        self.lbl_step_counter = QLabel("Adım 1 / 6")
        self.lbl_step_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_step_counter.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.btn_next_step = QPushButton("Sonraki Adım ▶")
        self.btn_next_step.setObjectName("primaryBtn")
        self.btn_next_step.clicked.connect(self.next_step)

        nav_row.addWidget(self.btn_prev_step)
        nav_row.addStretch()
        nav_row.addWidget(self.lbl_step_counter)
        nav_row.addStretch()
        nav_row.addWidget(self.btn_next_step)
        card_layout.addLayout(nav_row)

        layout.addWidget(card, stretch=1)

    def _build_diagnostics_tab(self) -> None:
        layout = QVBoxLayout(self.tab_diag)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        top_bar = QHBoxLayout()
        lbl_info = QLabel(
            "Sistem Öz-Tanılama aracı ekran çözünürlüğünü, ekran yakalama bütünlüğünü, şablon dosyalarını ve yönetici yetkilerini doğrular."
        )
        lbl_info.setStyleSheet("color: #94a3b8; font-size: 12px;")
        top_bar.addWidget(lbl_info)
        top_bar.addStretch()

        self.btn_run_diag = QPushButton("Tanılamayı Çalıştır")
        self.btn_run_diag.setObjectName("primaryBtn")
        self.btn_run_diag.clicked.connect(self.run_diagnostics)
        top_bar.addWidget(self.btn_run_diag)

        self.btn_diag_calib = QPushButton("Görsel Şablon ve Kalibrasyon Sihirbazı...")
        self.btn_diag_calib.setObjectName("accentBtn")
        self.btn_diag_calib.clicked.connect(self.open_calibration_wizard)
        top_bar.addWidget(self.btn_diag_calib)
        layout.addLayout(top_bar)

        # Scrollable container for diagnostic check cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.diag_container = QWidget()
        self.diag_cards_layout = QVBoxLayout(self.diag_container)
        self.diag_cards_layout.setContentsMargins(4, 4, 4, 4)
        self.diag_cards_layout.setSpacing(10)
        scroll.setWidget(self.diag_container)
        layout.addWidget(scroll, stretch=1)

        self.lbl_diag_summary = QLabel("Tanılama durumu: Hazır")
        self.lbl_diag_summary.setStyleSheet(
            "font-weight: bold; color: #38bdf8; padding: 4px;"
        )
        layout.addWidget(self.lbl_diag_summary)

    def set_step_index(self, index: int) -> None:
        if 0 <= index < len(self.GUIDE_STEPS):
            self._current_step_idx = index
            self._update_step_view()

    def next_step(self) -> None:
        if self._current_step_idx < len(self.GUIDE_STEPS) - 1:
            self._current_step_idx += 1
            self._update_step_view()

    def prev_step(self) -> None:
        if self._current_step_idx > 0:
            self._current_step_idx -= 1
            self._update_step_view()

    def _load_frame_pixmap(self, frame_filename: str, title: str) -> QPixmap:
        """Loads a reference frame from video_frames/ or synthesizes an informative preview banner."""
        try:
            frame_path = self.frames_dir / frame_filename
            if frame_path.exists():
                img_bgr = imread_unicode(frame_path, cv2.IMREAD_COLOR)
                if img_bgr is not None and img_bgr.size > 0:
                    return numpy_bgr_to_qpixmap(img_bgr)
        except Exception as exc:
            logger.warning("Failed reading guide frame %s: %s", frame_filename, exc)

        # Fallback rendered pixmap if file is missing
        pix = QPixmap(640, 360)
        pix.fill(QColor("#11151c"))
        painter = QPainter(pix)
        painter.setPen(QColor("#38bdf8"))
        painter.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        painter.drawText(
            pix.rect(),
            Qt.AlignmentFlag.AlignCenter,
            f"{frame_filename}\n{title}",
        )
        painter.end()
        return pix

    def _update_step_view(self) -> None:
        try:
            step = self.GUIDE_STEPS[self._current_step_idx]
            self.lbl_step_title.setText(step["title"])
            self.lbl_step_desc.setText(step["description"])
            self.lbl_step_counter.setText(
                f"Adım {self._current_step_idx + 1} / {len(self.GUIDE_STEPS)} — {step['frame']}"
            )

            for i, btn in enumerate(self._step_buttons):
                is_curr = i == self._current_step_idx
                btn.setChecked(is_curr)
                if is_curr:
                    btn.setStyleSheet(
                        "background-color: #2563eb; color: #ffffff; border-color: #60a5fa;"
                    )
                else:
                    btn.setStyleSheet("")

            pix = self._load_frame_pixmap(step["frame"], step["title"])
            scaled = pix.scaled(
                self.lbl_step_image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.lbl_step_image.setPixmap(scaled)

            self.btn_prev_step.setEnabled(self._current_step_idx > 0)
            self.btn_next_step.setEnabled(
                self._current_step_idx < len(self.GUIDE_STEPS) - 1
            )
        except Exception as exc:
            logger.error("Failed updating guide step view: %s", exc)

    def _check_resolution_and_monitor(self) -> Dict[str, Any]:
        """1. Resolution & Monitor Check (1920x1080 / 2560x1440)."""
        try:
            width, height = 0, 0
            try:
                import mss

                with mss.mss() as sct:
                    mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                    width, height = int(mon["width"]), int(mon["height"])
            except Exception:
                from PyQt6.QtGui import QGuiApplication

                screen = QGuiApplication.primaryScreen()
                if screen is not None:
                    sz = screen.size()
                    width, height = sz.width(), sz.height()

            standard_resolutions = {(1920, 1080), (2560, 1440)}
            passed = (width, height) in standard_resolutions
            if passed:
                detail = (
                    f"Desteklenen ekran çözünürlüğü algılandı: {width}x{height} (16:9 standart)."
                )
            else:
                detail = (
                    f"Algılanan çözünürlük: {width}x{height}. Önerilen: 1920x1080 veya 2560x1440 "
                    f"(farklı çözünürlük kullanıyorsanız Kalibrasyon Sihirbazı ile şablonları kesin)."
                )
            return {
                "title": "1. Ekran Çözünürlüğü ve Monitör Kontrolü (1920x1080 / 2560x1440)",
                "passed": passed,
                "badge": "✔ GEÇTİ" if passed else "⚠ UYARI",
                "detail": detail,
            }
        except Exception as exc:
            logger.error("Resolution check error: %s", exc)
            return {
                "title": "1. Ekran Çözünürlüğü ve Monitör Kontrolü (1920x1080 / 2560x1440)",
                "passed": False,
                "badge": "⚠ UYARI",
                "detail": f"Birincil monitör çözünürlüğü sorgulanamadı: {exc}",
            }

    def _check_screen_capture(self) -> Dict[str, Any]:
        """2. Screen Capture Check (Black screen / Fullscreen Exclusive test)."""
        try:
            frame_bgr: Optional[np.ndarray] = None
            try:
                import mss

                with mss.mss() as sct:
                    mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                    raw = np.array(sct.grab(mon))
                    frame_bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            except Exception:
                from PyQt6.QtGui import QGuiApplication

                screen = QGuiApplication.primaryScreen()
                if screen is not None:
                    qpix = screen.grabWindow(0)
                    qimg = qpix.toImage().convertToFormat(QImage.Format.Format_RGB888)
                    w, h = qimg.width(), qimg.height()
                    ptr = qimg.bits()
                    if ptr is not None and w > 0 and h > 0:
                        ptr.setsize(h * qimg.bytesPerLine())
                        arr = np.frombuffer(ptr, np.uint8).reshape((h, qimg.bytesPerLine()))
                        rgb = arr[:, : w * 3].reshape((h, w, 3))
                        frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if frame_bgr is None or frame_bgr.size == 0:
                return {
                    "title": "2. Ekran Yakalama Kontrolü (Siyah Ekran / Tam Ekran Özel Mod Testi)",
                    "passed": False,
                    "badge": "⚠ UYARI",
                    "detail": "Ekran yakalama boş döndü. Oyunu Çerçevesiz Pencere (Borderless Windowed) moduna alın.",
                }

            mean_lum = float(np.mean(frame_bgr))
            std_lum = float(np.std(frame_bgr))
            is_black = mean_lum < 2.0 and std_lum < 2.0

            if not is_black:
                return {
                    "title": "2. Ekran Yakalama Kontrolü (Siyah Ekran / Tam Ekran Özel Mod Testi)",
                    "passed": True,
                    "badge": "✔ GEÇTİ",
                    "detail": (
                        f"Canlı ekran görüntüsü sağlıklı yakalandı ({frame_bgr.shape[1]}x{frame_bgr.shape[0]}, "
                        f"ortalama parlaklık={mean_lum:.1f}). Tam Ekran engeli yok."
                    ),
                }
            return {
                "title": "2. Ekran Yakalama Kontrolü (Siyah Ekran / Tam Ekran Özel Mod Testi)",
                "passed": False,
                "badge": "⚠ UYARI",
                "detail": (
                    "Yakalanan kare tamamen siyah görünüyor (ortalama parlaklık < 2.0). "
                    "Oyunun 'Tam Ekran' yerine 'Çerçevesiz Pencere' (Borderless Windowed) modunda olduğundan emin olun."
                ),
            }
        except Exception as exc:
            logger.error("Screen capture diagnostic failed: %s", exc)
            return {
                "title": "2. Ekran Yakalama Kontrolü (Siyah Ekran / Tam Ekran Özel Mod Testi)",
                "passed": False,
                "badge": "⚠ UYARI",
                "detail": f"Ekran yakalama testi sırasında hata oluştu: {exc}",
            }

    def _check_template_files(self) -> Dict[str, Any]:
        """3. Template Files Check (templates/f_gather_template.png existence and size check)."""
        try:
            fg_path = self.templates_dir / "f_gather_template.png"
            if not fg_path.exists():
                return {
                    "title": "3. Şablon Dosyaları Kontrolü (templates/f_gather_template.png)",
                    "passed": False,
                    "badge": "⚠ UYARI",
                    "detail": (
                        f"{self.templates_dir} dizininde '{fg_path.name}' bulunamadı. "
                        "Şablon kesmek ve kaydetmek için 'Görsel Şablon ve Kalibrasyon Sihirbazı...' butonuna tıklayın."
                    ),
                }

            file_size = fg_path.stat().st_size
            img = imread_unicode(fg_path, cv2.IMREAD_COLOR)
            if img is None or img.size == 0 or file_size < 64:
                return {
                    "title": "3. Şablon Dosyaları Kontrolü (templates/f_gather_template.png)",
                    "passed": False,
                    "badge": "⚠ UYARI",
                    "detail": f"'{fg_path.name}' şablon dosyası bozuk veya boş ({file_size} bayt).",
                }

            h, w = img.shape[:2]
            passed = w >= 8 and h >= 8
            return {
                "title": "3. Şablon Dosyaları Kontrolü (templates/f_gather_template.png)",
                "passed": passed,
                "badge": "✔ GEÇTİ" if passed else "⚠ UYARI",
                "detail": (
                    f"'{fg_path.name}' ({w}x{h} piksel, {file_size} bayt) doğrulandı ve eşleşmeye hazır."
                ),
            }
        except Exception as exc:
            logger.error("Template file check error: %s", exc)
            return {
                "title": "3. Şablon Dosyaları Kontrolü (templates/f_gather_template.png)",
                "passed": False,
                "badge": "⚠ UYARI",
                "detail": f"Şablon dosyaları kontrol edilirken hata oluştu: {exc}",
            }

    def _check_admin_and_input_driver(self) -> Dict[str, Any]:
        """4. Administrator Privileges (ctypes.windll.shell32.IsUserAnAdmin()) & Input Driver Check."""
        try:
            is_admin = False
            try:
                if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "shell32"):
                    is_admin = bool(ctypes.windll.shell32.IsUserAnAdmin())
            except Exception as adm_exc:
                logger.debug("IsUserAnAdmin check error: %s", adm_exc)

            # Check Win32 SendInput availability via ctypes.windll.user32
            has_send_input = False
            try:
                if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "user32"):
                    has_send_input = hasattr(ctypes.windll.user32, "SendInput")
            except Exception:
                has_send_input = False

            if is_admin and has_send_input:
                return {
                    "title": "4. Yönetici Yetkileri ve Girdi Sürücüsü Kontrolü",
                    "passed": True,
                    "badge": "✔ GEÇTİ",
                    "detail": (
                        "Yönetici yetkileriyle çalışıyor (IsUserAnAdmin=True) ve Win32 SendInput sürücüsü aktif."
                    ),
                }

            reasons = []
            if not is_admin:
                reasons.append(
                    "Uygulama Yönetici olarak çalışmıyor (IsUserAnAdmin=False) — oyun tuş vuruşlarını algılamazsa uygulamayı Yönetici olarak çalıştırın"
                )
            if not has_send_input:
                reasons.append("Win32 user32.SendInput sürücüsü algılanamadı")

            return {
                "title": "4. Yönetici Yetkileri ve Girdi Sürücüsü Kontrolü",
                "passed": False,
                "badge": "⚠ UYARI",
                "detail": "; ".join(reasons) + ".",
            }
        except Exception as exc:
            logger.error("Admin & input driver check error: %s", exc)
            return {
                "title": "4. Yönetici Yetkileri ve Girdi Sürücüsü Kontrolü",
                "passed": False,
                "badge": "⚠ UYARI",
                "detail": f"Yetkiler doğrulanamadı: {exc}",
            }

    def run_diagnostics(self) -> List[Dict[str, Any]]:
        """Executes all 4 automated system diagnostic checks and renders PASS/WARN cards."""
        try:
            self._diagnostic_results = [
                self._check_resolution_and_monitor(),
                self._check_screen_capture(),
                self._check_template_files(),
                self._check_admin_and_input_driver(),
            ]

            # Clear old cards
            while self.diag_cards_layout.count():
                item = self.diag_cards_layout.takeAt(0)
                if item is not None:
                    w = item.widget()
                    if w is not None:
                        w.deleteLater()

            pass_count = 0
            for item in self._diagnostic_results:
                passed = bool(item.get("passed", False))
                if passed:
                    pass_count += 1
                card = self._create_diagnostic_card(item)
                self.diag_cards_layout.addWidget(card)

            self.diag_cards_layout.addStretch()

            total = len(self._diagnostic_results)
            if pass_count == total:
                self.lbl_diag_summary.setText(
                    f"Tanılama Özeti: ✔ Tüm {total}/{total} kontrol başarıyla geçti!"
                )
                self.lbl_diag_summary.setStyleSheet(
                    "font-weight: bold; color: #2ecc71; padding: 4px;"
                )
            else:
                warn_count = total - pass_count
                self.lbl_diag_summary.setText(
                    f"Tanılama Özeti: {pass_count}/{total} geçti, {warn_count} uyarı — ⚠ UYARI işaretli maddeleri inceleyin."
                )
                self.lbl_diag_summary.setStyleSheet(
                    "font-weight: bold; color: #f59e0b; padding: 4px;"
                )

            return self._diagnostic_results
        except Exception as exc:
            logger.error("Failed running diagnostics: %s", exc)
            return []

    def _create_diagnostic_card(self, item: Dict[str, Any]) -> QFrame:
        passed = bool(item.get("passed", False))
        badge_text = str(item.get("badge", "✔ GEÇTİ" if passed else "⚠ UYARI"))
        title_text = str(item.get("title", "Tanılama Kontrolü"))
        detail_text = str(item.get("detail", ""))

        card = QFrame()
        border_col = "#22c55e" if passed else "#f59e0b"
        card.setStyleSheet(
            f"background-color: #141822; border: 1px solid #2b3245; "
            f"border-left: 5px solid {border_col}; border-radius: 6px;"
        )

        row = QHBoxLayout(card)
        row.setContentsMargins(14, 12, 14, 12)
        row.setSpacing(14)

        badge_lbl = QLabel(badge_text)
        badge_lbl.setFixedWidth(90)
        badge_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if passed:
            badge_lbl.setStyleSheet(
                "background-color: rgba(34, 197, 94, 0.18); color: #4ade80; "
                "border: 1px solid #22c55e; border-radius: 4px; padding: 6px; font-weight: bold;"
            )
        else:
            badge_lbl.setStyleSheet(
                "background-color: rgba(245, 158, 11, 0.18); color: #fbbf24; "
                "border: 1px solid #f59e0b; border-radius: 4px; padding: 6px; font-weight: bold;"
            )
        row.addWidget(badge_lbl)

        text_col = QVBoxLayout()
        text_col.setSpacing(4)
        lbl_t = QLabel(title_text)
        lbl_t.setStyleSheet("font-weight: bold; font-size: 14px; color: #f1f5f9;")
        lbl_d = QLabel(detail_text)
        lbl_d.setWordWrap(True)
        lbl_d.setStyleSheet("color: #94a3b8; font-size: 12px;")
        text_col.addWidget(lbl_t)
        text_col.addWidget(lbl_d)
        row.addLayout(text_col, stretch=1)

        return card

    def open_calibration_wizard(self) -> None:
        """Opens the CalibrationWizard dialog modally and re-runs diagnostics if templates change."""
        try:
            self.calibration_requested.emit()
            wizard = CalibrationWizard(parent=self, base_dir=self.base_dir)
            wizard.template_updated.connect(lambda _: self.run_diagnostics())
            wizard.exec()
            self.run_diagnostics()
        except Exception as exc:
            logger.error("Failed opening CalibrationWizard from GuideDialog: %s", exc)

    @staticmethod
    def get_enterprise_modules_guide() -> List[Dict[str, str]]:
        """
        Returns a Turkish summary list of the 11 integrated enterprise & vision modules
        and the RTX YOLO Dataset Factory.
        """
        return [
            {
                "module": "1. F9 Ana Kontrol ve Yüzer Menü (F9 Menu)",
                "summary": "Gerçek zamanlı durum takibi, bot başlatma/durdurma, mod seçimi ve hızlı kılavuz erişimi sağlayan merkezi kontrol paneli."
            },
            {
                "module": "2. Görsel Şablon ve Kalibrasyon Sihirbazı (Calibration Wizard)",
                "summary": "F/E toplama butonu ve minimap ok şablonlarını özel ekran çözünürlükleri ve arayüz ölçekleri için kesip kalibre etmeyi sağlar."
            },
            {
                "module": "3. Otomatik Sistem Tanılama (System Diagnostics)",
                "summary": "Ekran çözünürlüğü, ekran yakalama bütünlüğü, şablon dosyaları ve Windows yönetici/SendInput yetkilerini otomatik doğrular."
            },
            {
                "module": "4. Mobil Uzaktan Bağlantı & Komuta Merkezi (Mobile Connect Dialog)",
                "summary": "Mobil cihazlar veya harici ağ üzerinden Gatherion-Local botunu uzaktan izleme, komut gönderme ve durum raporu alma modülü."
            },
            {
                "module": "5. RTX YOLO Veri Seti Fabrikası (RTX YOLO Dataset Factory)",
                "summary": "Yüksek performanslı RTX donanımı kullanarak otomatik veri toplama, etiketleme ve YOLO nesne tespiti model eğitimi sağlayan fabrika arayüzü."
            },
            {
                "module": "6. Hibrit Otomatik Haritalama ve Navigasyon (Auto Cartographer)",
                "summary": "Sağ üst minimap ikonları ile 3B dünya beyaz kanatlı toplama amblemlerini birleştirerek hatasız rota kaydı ve takibi yapar."
            },
            {
                "module": "7. Savaş Kurtarma ve Ölüm Yönetimi (Combat Recovery)",
                "summary": "Canavar saldırısı veya ölüm durumlarında güvenli bölgeye kaçış, otomatik dirilme (revive) ve rotaya geri dönme kurtarma sistemi."
            },
            {
                "module": "8. Harita ve Konum Görselleştiricisi (Map Viewer)",
                "summary": "Kayıtlı toplama rotalarını, ara noktaları (waypoints) ve materyal yoğunluk haritalarını 2B/3B olarak görselleştirip düzenleme aracı."
            },
            {
                "module": "9. Davranışsal Güvenlik ve Anti-Ban Koruyucusu (Behavioral Guard)",
                "summary": "Doğal insan benzeri fare hareketleri, rastgele gecikmeler ve anti-cheat önlemleriyle bot tespitini engelleyen koruma modülü."
            },
            {
                "module": "10. Gelişmiş Ayarlar ve Profil Yöneticisi (Settings Window)",
                "summary": "Oyun yolu, toplama tercihleri, performans ayarları ve kullanıcı profillerinin yönetildiği kapsamlı konfigürasyon merkezi."
            },
            {
                "module": "11. Akıllı Radar ve Çevresel Farkındalık (Radar)",
                "summary": "Çevredeki oyuncuları, canavarları ve toplama materyallerini gerçek zamanlı tarayıp ekranda radar HUD olarak gösteren modül."
            },
            {
                "module": "12. Canlı HUD ve Durum Gözlemcisi (Overlay HUD)",
                "summary": "Oyun ekranı üzerinde şeffaf katman olarak çalışan; anlık durum, istatistikler ve bot performansını gösteren heads-up display."
            }
        ]

    def get_guide_dialog_health_summary(self) -> dict:
        """Returns health summary of the guide dialog safely."""
        try:
            return {"guide_dialog_ready": True}
        except Exception:
            return {"guide_dialog_ready": False}

    def get_guide_dialog_ready_flag(self) -> dict:
        return {"module": "guide_dialog", "ready": True}

    def get_guide_dialog_version_info(self) -> dict:
        return {"module": "guide_dialog", "version": "2.0.0", "ready": True}

    def get_guide_dialog_diagnostic_snapshot(self) -> dict:
        return {"module": "guide_dialog", "status": "ok", "ready": True}


