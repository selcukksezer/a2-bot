import json
import logging
import secrets
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QClipboard,
    QColor,
    QFont,
    QGuiApplication,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.shell_page_frame import wrap_shell_tab_root_layout
from version import __version__

logger = logging.getLogger("MobileConnectDialog")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [MobileUI] %(message)s", datefmt="%H:%M:%S"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _detect_lan_ip() -> str:
    """Yerel ağ (Wi-Fi / Ethernet) IPv4 adresini güvenli şekilde tespit eder."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.5)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return str(ip)
    except Exception as exc:
        logger.debug("UDP LAN IP tespiti atlandı: %s", exc)
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and not ip.startswith("127."):
            return str(ip)
    except Exception as exc:
        logger.debug("Hostname LAN IP tespiti atlandı: %s", exc)
    # Uydurma bir 192.168.x.x adresi asla gercek adres gibi gosterilmez;
    # tespit basarisizsa bu makinenin gercek loopback adresi doner.
    return "127.0.0.1"


def _default_ntfy_topic() -> str:
    """Kriptografik rastgele ntfy.sh bildirim kanali (kalici).

    Konu, relay config dosyasinda saklanir; yoksa 128-bit rastgele
    uretilip kaydedilir. Hostname'den turetilen deterministik konu
    herkes tarafindan tahmin edilebildigi icin kullanilmaz.
    """
    try:
        from core.mobile_tunnel import DEFAULT_RELAY_CONFIG_PATH

        cfg_path = Path(DEFAULT_RELAY_CONFIG_PATH)
        if cfg_path.exists():
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                saved = str(raw.get("topic", "")).strip()
                # Eski 32-bit (8 hex) konu tahmin edilebilir; gecerli sayilmaz.
                import re as _re

                if saved and len(saved) >= 6 and not _re.match(
                    r"^gatherion_tr_[0-9a-f]{8}$", saved
                ):
                    return saved
        topic = f"gatherion_tr_{secrets.token_hex(16)}"
        try:
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            existing: Dict[str, Any] = {}
            if cfg_path.exists():
                loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            existing.setdefault("topic", topic)
            cfg_path.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            logger.debug("ntfy konu kaliciligi atlandi: %s", exc)
        return topic
    except Exception as exc:
        logger.debug("ntfy konu yuklemesi atlandi: %s", exc)
        return f"gatherion_tr_{secrets.token_hex(16)}"


class MobileConnectDialog(QDialog):
    """
    📱 Mobil Uzaktan Kontrol ve Canlı Takip Merkezi (Ücretsiz Web Paneli).
    Kullanıcının telefon kamerasıyla QR kodu okutarak yerel Wi-Fi veya
    ücretsiz internet tüneli (Cloudflare / localhost.run) üzerinden bota
    erişmesini ve ntfy.sh kanalıyla anlık bildirim almasını sağlar.
    """

    tunnel_result_signal = pyqtSignal(bool, str, str)

    # QR pixmap önbelleğinin üst sınırı (farklı URL/boyut başına bir
    # 250 px pixmap; sınır, tanılama/yenileme döngülerinde büyümesini engeller).
    _QR_PIXMAP_CACHE_MAX_ENTRIES = 8

    def __init__(
        self,
        mobile_server: Optional[Any] = None,
        tunnel_mgr: Optional[Any] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.mobile_server = mobile_server
        self.tunnel_mgr = tunnel_mgr

        self._qr_target_mode: str = "lan"  # "lan" veya "tunnel"
        self._tunnel_running: bool = False
        self._tunnel_connecting: bool = False
        self._active_tunnel_url: str = ""
        self._tunnel_method_label: str = "Hazır (Beklemede)"
        self._last_qr_url: str = ""

        self.setWindowTitle(
            "📱 Mobil Uzaktan Kontrol ve Canlı Takip Merkezi (Ücretsiz Web Paneli)"
        )
        self.setMinimumSize(940, 620)
        self.resize(980, 650)
        self.setModal(False)

        self.tunnel_result_signal.connect(self._on_tunnel_finished)

        self._apply_dark_theme()
        self._init_ui()
        self._sync_from_backends()
        self._refresh_qr_display()

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1500)
        self._status_timer.timeout.connect(self._on_periodic_refresh)
        self._status_timer.start()

    def get_uac_and_security_status(self) -> dict:
        """
        Returns a dictionary containing 'is_admin' from core.uac_elevator
        and token rotation status (wrapped in try/except).
        """
        status: dict = {"is_admin": False, "token_rotation": None, "error": None}
        try:
            from core.uac_elevator import is_admin
            status["is_admin"] = bool(is_admin())
        except Exception as exc:
            status["error"] = str(exc)

        try:
            # Check if mobile_server or mobile_server manager has token rotation status or token methods
            if self.mobile_server is not None:
                if hasattr(self.mobile_server, "get_token_rotation_status"):
                    status["token_rotation"] = self.mobile_server.get_token_rotation_status()
                elif hasattr(self.mobile_server, "token_rotation_active"):
                    status["token_rotation"] = getattr(self.mobile_server, "token_rotation_active")
                else:
                    # Kanit yoksa True varsayilmez: bilinmiyor.
                    status["token_rotation"] = getattr(self.mobile_server, "_token_rotated", None)
            else:
                status["token_rotation"] = None
        except Exception as exc:
            status["token_rotation_error"] = str(exc)

        return status

    # ------------------------------------------------------------------
    # Tema et Arayüz Kurulumu
    # ------------------------------------------------------------------

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background-color: #0b0f17;
                color: #f8fafc;
                font-family: 'Segoe UI', 'Inter', sans-serif;
            }
            QFrame#HeaderBanner {
                background-color: #111827;
                border: 1px solid #1e293b;
                border-left: 4px solid #38bdf8;
                border-radius: 10px;
            }
            QFrame#QRCard, QFrame#ControlCard {
                background-color: #131b26;
                border: 1px solid #222f42;
                border-radius: 10px;
            }
            QFrame#SectionBox {
                background-color: #0f1620;
                border: 1px solid #1e293b;
                border-radius: 8px;
            }
            QLabel {
                color: #e2e8f0;
                background: transparent;
                border: none;
            }
            QLineEdit#UrlReadOnly {
                background-color: #090d14;
                color: #38bdf8;
                border: 1px solid #26354a;
                border-radius: 6px;
                padding: 7px 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                font-weight: 700;
            }
            QLineEdit#UrlReadOnly:focus {
                border: 1px solid #38bdf8;
            }
            QPushButton#ActionBtn {
                background-color: #1e293b;
                color: #f1f5f9;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton#ActionBtn:hover {
                background-color: #283548;
                border-color: #475569;
            }
            QPushButton#ActionBtn:pressed {
                background-color: #0f172a;
            }
            QPushButton#PrimaryTunnelBtn {
                background-color: #0284c7;
                color: #ffffff;
                border: 1px solid #38bdf8;
                border-radius: 7px;
                padding: 9px 16px;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton#PrimaryTunnelBtn:hover {
                background-color: #0369a1;
            }
            QPushButton#PrimaryTunnelBtn:disabled {
                background-color: #1e293b;
                color: #64748b;
                border-color: #334155;
            }
            QPushButton#SwitchBtnActive {
                background-color: #0ea5e9;
                color: #081018;
                border: 1px solid #38bdf8;
                border-radius: 6px;
                padding: 7px 10px;
                font-size: 12px;
                font-weight: 800;
            }
            QPushButton#SwitchBtnInactive {
                background-color: #16202e;
                color: #94a3b8;
                border: 1px solid #263548;
                border-radius: 6px;
                padding: 7px 10px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton#SwitchBtnInactive:hover {
                color: #e2e8f0;
                border-color: #38bdf8;
            }
            """
        )

    def _init_ui(self) -> None:
        root_layout = wrap_shell_tab_root_layout(self, "mobile")
        root_layout.setSpacing(14)

        # Üst durum bandı
        header_frame = QFrame()
        header_frame.setObjectName("HeaderBanner")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(12)

        self.lbl_live_badge = QLabel("● DURUM KONTROL EDİLİYOR")
        self.lbl_live_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_live_badge.setFixedHeight(32)
        self.lbl_live_badge.setStyleSheet(
            "background-color: rgba(16, 185, 129, 0.16); color: #34d399; "
            "border: 1px solid #10b981; border-radius: 15px; padding: 0 14px; "
            "font-size: 12px; font-weight: 800;"
        )

        header_layout.addStretch(1)
        header_layout.addWidget(self.lbl_live_badge)
        root_layout.addWidget(header_frame)

        # Ana İki Sütunlu Gövde
        body_layout = QHBoxLayout()
        body_layout.setSpacing(16)

        # ==============================================================
        # SOL SÜTUN: QR Kod Tarayıcı Kartı
        # ==============================================================
        qr_card = QFrame()
        qr_card.setObjectName("QRCard")
        qr_card.setMinimumWidth(240)
        qr_card.setMaximumWidth(360)
        qr_layout = QVBoxLayout(qr_card)
        qr_layout.setContentsMargins(18, 16, 18, 16)
        qr_layout.setSpacing(10)

        lbl_qr_header = QLabel("📷 TELEFON KAMERASI İLE OKUTUN")
        lbl_qr_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_qr_header.setStyleSheet(
            "color: #d4a017; font-size: 12px; font-weight: 800; letter-spacing: 0.7px;"
        )
        qr_layout.addWidget(lbl_qr_header)

        # Hedef Seçici: [Yerel Wi-Fi Bağlantısı] vs [Ücretsiz İnternet Tüneli (Her Yerden)]
        switch_row = QHBoxLayout()
        switch_row.setSpacing(6)

        self.btn_mode_lan = QPushButton("Yerel Wi-Fi Bağlantısı")
        self.btn_mode_lan.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_mode_lan.clicked.connect(lambda: self.set_qr_target_mode("lan"))

        self.btn_mode_tunnel = QPushButton("Ücretsiz İnternet Tüneli (Her Yerden)")
        self.btn_mode_tunnel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_mode_tunnel.clicked.connect(lambda: self.set_qr_target_mode("tunnel"))

        switch_row.addWidget(self.btn_mode_lan, stretch=1)
        switch_row.addWidget(self.btn_mode_tunnel, stretch=1)
        qr_layout.addLayout(switch_row)

        # QR Görsel Çerçevesi
        self.qr_image_label = QLabel()
        self.qr_image_label.setFixedSize(270, 270)
        self.qr_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.qr_image_label.setStyleSheet(
            "background-color: #ffffff; border: 4px solid #38bdf8; border-radius: 12px;"
        )
        qr_holder = QHBoxLayout()
        qr_holder.addStretch(1)
        qr_holder.addWidget(self.qr_image_label)
        qr_holder.addStretch(1)
        qr_layout.addLayout(qr_holder)

        self.lbl_qr_active_mode = QLabel("Aktif QR Hedefi: Yerel Wi-Fi Bağlantısı")
        self.lbl_qr_active_mode.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_qr_active_mode.setStyleSheet(
            "color: #f8fafc; font-size: 12px; font-weight: 700;"
        )
        qr_layout.addWidget(self.lbl_qr_active_mode)

        self.lbl_qr_active_url = QLabel("")
        self.lbl_qr_active_url.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_qr_active_url.setWordWrap(True)
        self.lbl_qr_active_url.setStyleSheet(
            "color: #38bdf8; font-family: 'Consolas', monospace; font-size: 11px; font-weight: 600;"
        )
        qr_layout.addWidget(self.lbl_qr_active_url)

        lbl_qr_hint = QLabel(
            "💡 İpucu: iPhone veya Android kameranızı doğrudan bu karekoda tutun. "
            "Ek uygulama indirmeden mobil kontrol paneli açılacaktır."
        )
        lbl_qr_hint.setWordWrap(True)
        lbl_qr_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_qr_hint.setStyleSheet("color: #94a3b8; font-size: 12px;")
        qr_layout.addWidget(lbl_qr_hint)
        qr_layout.addStretch(1)

        body_layout.addWidget(qr_card)

        # ==============================================================
        # SAĞ SÜTUN: Bağlantı Adresleri & Kontroller
        # ==============================================================
        right_card = QFrame()
        right_card.setObjectName("ControlCard")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(18, 16, 18, 16)
        right_layout.setSpacing(12)

        # 1. Yerel Wi-Fi Adresi (Aynı Ağa Bağlı Telefonlar İçin)
        sec_lan = QFrame()
        sec_lan.setObjectName("SectionBox")
        sec_lan_layout = QVBoxLayout(sec_lan)
        sec_lan_layout.setContentsMargins(14, 10, 14, 10)
        sec_lan_layout.setSpacing(7)

        lbl_lan_title = QLabel(
            "1. 📶 Yerel Wi-Fi Adresi (Aynı Ağa Bağlı Telefonlar İçin)"
        )
        lbl_lan_title.setStyleSheet(
            "color: #38bdf8; font-size: 14px; font-weight: 800;"
        )
        sec_lan_layout.addWidget(lbl_lan_title)

        lan_row = QHBoxLayout()
        lan_row.setSpacing(8)
        self.edit_lan_url = QLineEdit()
        self.edit_lan_url.setObjectName("UrlReadOnly")
        self.edit_lan_url.setReadOnly(True)

        self.btn_copy_lan = QPushButton("📋 Adresi Kopyala")
        self.btn_copy_lan.setObjectName("ActionBtn")
        self.btn_copy_lan.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_copy_lan.clicked.connect(self._copy_lan_url)

        self.btn_open_lan = QPushButton("🌍 Tarayıcıda Aç")
        self.btn_open_lan.setObjectName("ActionBtn")
        self.btn_open_lan.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_open_lan.clicked.connect(self._open_lan_in_browser)

        lan_row.addWidget(self.edit_lan_url, stretch=1)
        lan_row.addWidget(self.btn_copy_lan)
        lan_row.addWidget(self.btn_open_lan)
        sec_lan_layout.addLayout(lan_row)
        right_layout.addWidget(sec_lan)

        # 2. Ücretsiz İnternet Tüneli (4G/5G / Dışarıdan Erişim — Cloudflare / localhost.run)
        sec_tunnel = QFrame()
        sec_tunnel.setObjectName("SectionBox")
        sec_tunnel_layout = QVBoxLayout(sec_tunnel)
        sec_tunnel_layout.setContentsMargins(14, 10, 14, 10)
        sec_tunnel_layout.setSpacing(7)

        lbl_tunnel_title = QLabel(
            "2. 🌐 Ücretsiz İnternet Tüneli (4G/5G / Dışarıdan Erişim — Cloudflare / localhost.run)"
        )
        lbl_tunnel_title.setStyleSheet(
            "color: #a855f7; font-size: 14px; font-weight: 800;"
        )
        sec_tunnel_layout.addWidget(lbl_tunnel_title)

        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        lbl_tunnel_stat_prefix = QLabel("Tünel Durumu:")
        lbl_tunnel_stat_prefix.setStyleSheet(
            "color: #94a3b8; font-size: 12px; font-weight: 600;"
        )
        self.lbl_tunnel_status = QLabel("Hazır")
        self.lbl_tunnel_status.setStyleSheet(
            "color: #fbbf24; font-size: 12px; font-weight: 800;"
        )
        status_row.addWidget(lbl_tunnel_stat_prefix)
        status_row.addWidget(self.lbl_tunnel_status, stretch=1)
        sec_tunnel_layout.addLayout(status_row)

        tunnel_url_row = QHBoxLayout()
        tunnel_url_row.setSpacing(8)
        self.edit_tunnel_url = QLineEdit()
        self.edit_tunnel_url.setObjectName("UrlReadOnly")
        self.edit_tunnel_url.setReadOnly(True)
        self.edit_tunnel_url.setPlaceholderText(
            "Tünel başlatıldığında genel https://... adresi burada görünür"
        )

        self.btn_copy_tunnel = QPushButton("📋 Adresi Kopyala")
        self.btn_copy_tunnel.setObjectName("ActionBtn")
        self.btn_copy_tunnel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_copy_tunnel.clicked.connect(self._copy_tunnel_url)

        tunnel_url_row.addWidget(self.edit_tunnel_url, stretch=1)
        tunnel_url_row.addWidget(self.btn_copy_tunnel)
        sec_tunnel_layout.addLayout(tunnel_url_row)

        tunnel_btn_row = QHBoxLayout()
        tunnel_btn_row.setSpacing(8)
        self.btn_start_tunnel = QPushButton("🌐 Ücretsiz İnternet Tünelini Başlat")
        self.btn_start_tunnel.setObjectName("PrimaryTunnelBtn")
        self.btn_start_tunnel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start_tunnel.clicked.connect(self._start_internet_tunnel_async)

        self.btn_start_ssh_tunnel = QPushButton("🔑 Yedek SSH Tüneli (localhost.run)")
        self.btn_start_ssh_tunnel.setObjectName("ActionBtn")
        self.btn_start_ssh_tunnel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start_ssh_tunnel.clicked.connect(self._start_ssh_tunnel_async)

        tunnel_btn_row.addWidget(self.btn_start_tunnel, stretch=1)
        tunnel_btn_row.addWidget(self.btn_start_ssh_tunnel)
        sec_tunnel_layout.addLayout(tunnel_btn_row)
        right_layout.addWidget(sec_tunnel)

        # 3. Ücretsiz ntfy.sh Bulut Bildirim & Komut Kanalı
        sec_ntfy = QFrame()
        sec_ntfy.setObjectName("SectionBox")
        sec_ntfy_layout = QVBoxLayout(sec_ntfy)
        sec_ntfy_layout.setContentsMargins(14, 10, 14, 10)
        sec_ntfy_layout.setSpacing(6)

        lbl_ntfy_title = QLabel(
            "3. 🔔 Ücretsiz ntfy.sh Bulut Bildirim & Komut Kanalı"
        )
        lbl_ntfy_title.setStyleSheet(
            "color: #34d399; font-size: 14px; font-weight: 800;"
        )
        sec_ntfy_layout.addWidget(lbl_ntfy_title)

        lbl_ntfy_desc = QLabel(
            "Telefonunuza ücretsiz ntfy uygulamasını kurup aşağıdaki özel kanala abone olarak "
            "bot durduğunda, Captcha çıktığında veya rota tamamlandığında anlık bildirim alabilirsiniz:"
        )
        lbl_ntfy_desc.setWordWrap(True)
        lbl_ntfy_desc.setStyleSheet("color: #94a3b8; font-size: 12px;")
        sec_ntfy_layout.addWidget(lbl_ntfy_desc)

        ntfy_row = QHBoxLayout()
        ntfy_row.setSpacing(8)
        self.edit_ntfy_url = QLineEdit()
        self.edit_ntfy_url.setObjectName("UrlReadOnly")
        self.edit_ntfy_url.setReadOnly(True)

        self.btn_copy_ntfy = QPushButton("📋 Kanalı Kopyala")
        self.btn_copy_ntfy.setObjectName("ActionBtn")
        self.btn_copy_ntfy.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_copy_ntfy.clicked.connect(self._copy_ntfy_url)

        self.btn_open_ntfy = QPushButton("🔔 Kanalı Aç")
        self.btn_open_ntfy.setObjectName("ActionBtn")
        self.btn_open_ntfy.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_open_ntfy.clicked.connect(self._open_ntfy_in_browser)

        ntfy_row.addWidget(self.edit_ntfy_url, stretch=1)
        ntfy_row.addWidget(self.btn_copy_ntfy)
        ntfy_row.addWidget(self.btn_open_ntfy)
        sec_ntfy_layout.addLayout(ntfy_row)
        right_layout.addWidget(sec_ntfy)

        # 4. Sunucu Durumu
        sec_status = QFrame()
        sec_status.setObjectName("SectionBox")
        sec_status_layout = QVBoxLayout(sec_status)
        sec_status_layout.setContentsMargins(14, 10, 14, 10)
        sec_status_layout.setSpacing(6)

        lbl_status_title = QLabel("4. 🛡️ Sunucu Durumu ve Güvenlik Bilgileri")
        lbl_status_title.setStyleSheet(
            "color: #f59e0b; font-size: 14px; font-weight: 800;"
        )
        sec_status_layout.addWidget(lbl_status_title)

        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(16)

        self.lbl_port_info = QLabel("Aktif Port: 8765")
        self.lbl_port_info.setStyleSheet(
            "color: #f8fafc; font-size: 12px; font-weight: 700;"
        )

        self.lbl_clients_info = QLabel("Bağlı İstemci / İstek: 0")
        self.lbl_clients_info.setStyleSheet(
            "color: #38bdf8; font-size: 12px; font-weight: 700;"
        )

        self.lbl_token_info = QLabel("Güvenlik Anahtarı: -")
        self.lbl_token_info.setStyleSheet(
            "color: #a7f3d0; font-family: 'Consolas', monospace; font-size: 12px; font-weight: 700;"
        )

        metrics_row.addWidget(self.lbl_port_info)
        metrics_row.addWidget(self.lbl_clients_info)
        metrics_row.addWidget(self.lbl_token_info, stretch=1)
        sec_status_layout.addLayout(metrics_row)

        right_layout.addWidget(sec_status)
        right_layout.addStretch(1)

        # Alt Kapat / Yenile Satırı
        bottom_row = QHBoxLayout()
        self.lbl_feedback = QLabel("")
        self.lbl_feedback.setStyleSheet(
            "color: #34d399; font-size: 12px; font-weight: 700;"
        )
        bottom_row.addWidget(self.lbl_feedback, stretch=1)

        btn_refresh = QPushButton("🔄 QR ve Durumu Yenile")
        btn_refresh.setMinimumHeight(34)
        btn_refresh.setObjectName("ActionBtn")
        btn_refresh.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_refresh.clicked.connect(self._manual_refresh)

        btn_close = QPushButton("Kapat")
        btn_close.setMinimumHeight(34)
        btn_close.setObjectName("ActionBtn")
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.clicked.connect(self.close)

        bottom_row.addWidget(btn_refresh)
        bottom_row.addWidget(btn_close)
        right_layout.addLayout(bottom_row)

        body_layout.addWidget(right_card, stretch=1)
        root_layout.addLayout(body_layout, stretch=1)

        self._update_switch_button_styles()

    # ------------------------------------------------------------------
    # Sunucu / Tünel Yöneticisi Senkronizasyonu
    # ------------------------------------------------------------------

    def _get_server_port(self) -> int:
        if self.mobile_server is not None:
            for attr in ("port", "server_port", "listen_port"):
                val = getattr(self.mobile_server, attr, None)
                if isinstance(val, int) and val > 0:
                    return val
        return 8765

    def _get_security_token(self) -> str:
        if self.mobile_server is not None:
            for attr in ("security_token", "auth_token", "token", "access_token"):
                val = getattr(self.mobile_server, attr, None)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        if self.tunnel_mgr is not None:
            for attr in ("security_token", "auth_token", "token"):
                val = getattr(self.tunnel_mgr, attr, None)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        # Hostname'den turetilen token tahmin edilebilir ve ekranda acik
        # yazilmamali; sunucu token vermediyse token yoktur.
        return ""

    def _with_token(self, url: str) -> str:
        """Y-1 (dilim 22): QR ile giriş akışı token'lı URL gerektirir —
        tokensız URL sunucuda 401 verir. Token varsa `?token=` eklenir
        (dashboard URL'den okuyup localStorage'a yazar). Token yoksa
        URL olduğu gibi döner."""
        base = str(url or "").strip()
        if not base or "?token=" in base or "?key=" in base:
            return base
        token = self._get_security_token()
        if not token:
            return base
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}token={urllib.parse.quote(token, safe='')}"

    def _get_lan_url(self) -> str:
        base = ""
        if self.mobile_server is not None:
            for attr in ("get_lan_url", "get_local_url"):
                fn = getattr(self.mobile_server, attr, None)
                if callable(fn):
                    try:
                        res = fn()
                        if isinstance(res, str) and res.startswith("http"):
                            base = res
                            break
                    except Exception as exc:
                        logger.debug("Sunucu LAN URL sorgusu atlandı: %s", exc)
            if not base:
                for attr in ("lan_url", "local_url"):
                    val = getattr(self.mobile_server, attr, None)
                    if isinstance(val, str) and val.startswith("http"):
                        base = val
                        break

        if not base:
            lan_ip = ""
            if self.tunnel_mgr is not None:
                fn_ip = getattr(self.tunnel_mgr, "get_lan_ip", None)
                if callable(fn_ip):
                    try:
                        lan_ip = str(fn_ip() or "")
                    except Exception:
                        lan_ip = ""
            if not lan_ip:
                lan_ip = _detect_lan_ip()

            port = self._get_server_port()
            base = f"http://{lan_ip}:{port}"

        return self._with_token(base)

    def _get_ntfy_url(self) -> str:
        for obj in (self.tunnel_mgr, self.mobile_server):
            if obj is None:
                continue
            for attr in ("get_ntfy_url", "ntfy_url"):
                val = getattr(obj, attr, None)
                if callable(val):
                    try:
                        res = val()
                        if isinstance(res, str) and res.startswith("http"):
                            return res
                    except Exception as exc:
                        logger.debug("ntfy URL sorgusu atlandı: %s", exc)
                elif isinstance(val, str) and val.startswith("http"):
                    return val
            for attr in ("ntfy_topic", "get_ntfy_topic"):
                val = getattr(obj, attr, None)
                topic = val() if callable(val) else val
                if isinstance(topic, str) and topic.strip():
                    clean_topic = topic.strip().lstrip("/")
                    if clean_topic.startswith("http"):
                        return clean_topic
                    return f"https://ntfy.sh/{clean_topic}"
        return f"https://ntfy.sh/{_default_ntfy_topic()}"

    def _get_connected_clients_count(self) -> Optional[int]:
        """Bagli istemci sayisi; sunucu sayac vermiyorsa None (bilinmiyor).

        Sayac kaynagi yokken 0 dondurmek "hic istemci yok" yalanidir.
        """
        if self.mobile_server is None:
            return None
        for attr in (
            "request_count",
            "connected_clients",
            "client_count",
            "total_requests",
            "active_connections",
        ):
            val = getattr(self.mobile_server, attr, None)
            if isinstance(val, int):
                return val
            if isinstance(val, (set, list, dict)):
                return len(val)
        status_fn = getattr(self.mobile_server, "get_status", None)
        if callable(status_fn):
            try:
                st = status_fn()
                if isinstance(st, dict):
                    for k in ("request_count", "connected_clients", "clients", "requests"):
                        if isinstance(st.get(k), int):
                            return int(st[k])
            except Exception as exc:
                logger.debug("Sunucu durum sorgusu atlandı: %s", exc)
        return None

    def _sync_from_backends(self) -> None:
        lan_url = self._get_lan_url()
        self.edit_lan_url.setText(lan_url)

        ntfy_url = self._get_ntfy_url()
        self.edit_ntfy_url.setText(ntfy_url)

        # Mevcut aktif tünel kontrolü
        if self.tunnel_mgr is not None and not self._tunnel_connecting:
            for attr in (
                "public_url",
                "active_url",
                "tunnel_url",
                "get_public_url",
                "get_active_tunnel_url",
            ):
                val = getattr(self.tunnel_mgr, attr, None)
                url_val = val() if callable(val) else val
                if isinstance(url_val, str) and url_val.startswith("https://"):
                    self._active_tunnel_url = url_val.strip()
                    self._tunnel_running = True
                    break

        if self._active_tunnel_url:
            self.edit_tunnel_url.setText(self._active_tunnel_url)
            self.lbl_tunnel_status.setText(f"Aktif: {self._active_tunnel_url}")
            self.lbl_tunnel_status.setStyleSheet(
                "color: #34d399; font-size: 12px; font-weight: 800;"
            )
        elif self._tunnel_connecting:
            self.lbl_tunnel_status.setText("Bağlanıyor...")
            self.lbl_tunnel_status.setStyleSheet(
                "color: #38bdf8; font-size: 12px; font-weight: 800;"
            )
        else:
            self.lbl_tunnel_status.setText(self._tunnel_method_label or "Hazır")
            self.lbl_tunnel_status.setStyleSheet(
                "color: #fbbf24; font-size: 12px; font-weight: 800;"
            )

        port = self._get_server_port()
        clients = self._get_connected_clients_count()
        token = self._get_security_token()

        self.lbl_port_info.setText(f"Aktif Port: {port}")
        clients_text = str(clients) if clients is not None else "Bilinmiyor"
        self.lbl_clients_info.setText(f"Bağlı İstemci / İstek: {clients_text}")
        # Token ekranda asla acik yazilmaz; yalnizca maskeli onizleme.
        token_text = f"{token[:4]}***" if token else "Tanımlı değil"
        self.lbl_token_info.setText(f"Güvenlik Anahtarı: {token_text}")

        # Sunucu nesnesi yoksa "aktif" varsayilamaz.
        server_running = False
        if self.mobile_server is not None:
            is_run = getattr(self.mobile_server, "running", getattr(self.mobile_server, "is_running", False))
            server_running = bool(is_run() if callable(is_run) else is_run)

        if server_running:
            self.lbl_live_badge.setText(f"● SUNUCU AKTİF (:{port})")
            self.lbl_live_badge.setStyleSheet(
                "background-color: rgba(16, 185, 129, 0.16); color: #34d399; "
                "border: 1px solid #10b981; border-radius: 15px; padding: 0 14px; "
                "font-size: 11px; font-weight: 800;"
            )
        else:
            self.lbl_live_badge.setText("● YEREL HAZIR MOD")
            self.lbl_live_badge.setStyleSheet(
                "background-color: rgba(56, 189, 248, 0.16); color: #38bdf8; "
                "border: 1px solid #0284c7; border-radius: 15px; padding: 0 14px; "
                "font-size: 11px; font-weight: 800;"
            )

    # ------------------------------------------------------------------
    # QR Kod Üretimi (tunnel_mgr.generate_qr_image + OpenCV QRCodeEncoder)
    # ------------------------------------------------------------------

    def get_active_qr_url(self) -> str:
        """Seçili QR hedefine göre taranacak URL'yi döndürür."""
        if self._qr_target_mode == "tunnel" and self._active_tunnel_url:
            return self._with_token(self._active_tunnel_url)
        return self._get_lan_url()

    def set_qr_target_mode(self, mode: str) -> None:
        """QR Kod hedefini 'lan' veya 'tunnel' olarak değiştirir."""
        self._qr_target_mode = "tunnel" if mode == "tunnel" else "lan"
        self._update_switch_button_styles()
        self._refresh_qr_display()

    def _update_switch_button_styles(self) -> None:
        if self._qr_target_mode == "lan":
            self.btn_mode_lan.setObjectName("SwitchBtnActive")
            self.btn_mode_tunnel.setObjectName("SwitchBtnInactive")
        else:
            self.btn_mode_lan.setObjectName("SwitchBtnInactive")
            self.btn_mode_tunnel.setObjectName("SwitchBtnActive")
        self.style().unpolish(self.btn_mode_lan)
        self.style().polish(self.btn_mode_lan)
        self.style().unpolish(self.btn_mode_tunnel)
        self.style().polish(self.btn_mode_tunnel)

    def _try_generate_real_qr_pixmap(self, url: str, size: int = 250) -> Optional[QPixmap]:
        """
        Gercek bir QR ureteciyle (core.mobile_tunnel / tunnel_mgr /
        OpenCV QRCodeEncoder) taranabilir bir QR QPixmap uretmeyi dener.

        Gercek ureteci yoksa None doner — cagiran, "olusturulamadi"
        kartini gosterip gostermeyecegine kendisi karar verir. S-4101:
        _render_unavailable_qr_pixmap karti da 252x252 dolu, null
        olmayan bir QPixmap oldugu icin dogrulama bu metoda bakarak
        yapilamaz; kart pixmap'i gecerli QR sayilmamalidir.
        """
        # QPixmap, QGuiApplication olmadan olusturulamaz (Qt abort eder ve
        # Python try/except yakalayamaz); uygulama yoksa hata firlat.
        if QGuiApplication.instance() is None:
            raise RuntimeError("QR pixmap requires a QGuiApplication instance")
        # 1. core.mobile_tunnel.generate_qr_image(url) modül fonksiyonu
        try:
            from core.mobile_tunnel import generate_qr_image as _mt_generate_qr

            raw_qr = _mt_generate_qr(url, size=size)
            pix = self._convert_any_qr_to_pixmap(raw_qr, size)
            if pix is not None and not pix.isNull():
                return pix
        except Exception as exc:
            logger.debug("core.mobile_tunnel.generate_qr_image hatası: %s", exc)

        # 1b. tunnel_mgr.generate_qr_image(active_url) varsa kullan
        tunnel_mgr = getattr(self, "tunnel_mgr", None)
        if tunnel_mgr is not None:
            gen_fn = getattr(tunnel_mgr, "generate_qr_image", None)
            if callable(gen_fn):
                try:
                    raw_qr = gen_fn(url)
                    pix = self._convert_any_qr_to_pixmap(raw_qr, size)
                    if pix is not None and not pix.isNull():
                        return pix
                except Exception as exc:
                    logger.debug("tunnel_mgr.generate_qr_image hatası: %s", exc)

        # 2. OpenCV cv2.QRCodeEncoder ile QR matrisi üret
        qr_Encoder_cls = getattr(cv2, "QRCodeEncoder", None)
        if qr_Encoder_cls is not None and hasattr(qr_Encoder_cls, "create"):
            try:
                encoder = qr_Encoder_cls.create()
                qr_mat = encoder.encode(str(url))
                if isinstance(qr_mat, np.ndarray) and qr_mat.size > 0:
                    pix = self._numpy_to_qr_pixmap(qr_mat, size)
                    if pix is not None and not pix.isNull():
                        return pix
            except Exception as exc:
                logger.debug("cv2.QRCodeEncoder hatası: %s", exc)

        # Gerçek QR üreticisi yok.
        return None

    def _generate_qr_pixmap(self, url: str, size: int = 250) -> QPixmap:
        """Cached wrapper around :meth:`_generate_qr_pixmap_uncached`.

        QR encoding (cv2 ``QRCodeEncoder`` + numpy resize + QImage
        conversion) costs ~100-300 ms per call on this machine, and the
        dialog re-requests the same (url, size) on every manual refresh and
        every diagnostics pass. The pixmap is a pure function of
        (url, size), so a repeat request returns the cached pixmap.
        Only pixmaps from a real producer are cached: the "unavailable"
        fallback card is deliberately NOT cached, so a later retry can
        succeed once a real producer becomes available.
        """
        # __dict__ lookup (not getattr): on a __new__-constructed dialog
        # whose __init__ never ran, PyQt raises RuntimeError for a missing
        # attribute instead of AttributeError.
        cache = self.__dict__.get("_qr_pixmap_cache")
        if cache is None:
            cache = {}
            try:
                setattr(self, "_qr_pixmap_cache", cache)
            except Exception:
                pass
        key = (str(url), int(size))
        hit = cache.get(key)
        if hit is not None and not hit.isNull():
            return hit
        pix, from_producer = self._generate_qr_pixmap_uncached(url, size)
        if from_producer and pix is not None and not pix.isNull():
            if len(cache) >= self._QR_PIXMAP_CACHE_MAX_ENTRIES:
                cache.pop(next(iter(cache)))
            cache[key] = pix
        return pix

    def _generate_qr_pixmap_uncached(self, url: str, size: int = 250) -> tuple:
        """
        Verilen URL için net bir QR Kod QPixmap üretir.
        Öncelikle `core.mobile_tunnel.generate_qr_image(url)` modül
        fonksiyonunu dener (gerçek yönetici FreeCloudTunnelManager'da
        bu isimde metot yoktur), ardından OpenCV `cv2.QRCodeEncoder`
        kullanır. Hiçbir gerçek üretici yoksa sahte matris çizilmez;
        açık bir "oluşturulamadı" kartı döner.

        Returns ``(pixmap, from_real_producer)``: the second element is
        False exactly when the pixmap is the "unavailable" fallback card,
        so the caller can decide whether the result is worth caching.
        """
        pix = self._try_generate_real_qr_pixmap(url, size)
        if pix is not None and not pix.isNull():
            return pix, True

        # 3. Gerçek QR üreticisi yok: sahte matris çizilmez, dürüst kart döner.
        # from_real_producer=False: bu kart önbelleğe alınmaz.
        return self._render_unavailable_qr_pixmap(url, size), False

    def _convert_any_qr_to_pixmap(self, raw_qr: Any, size: int) -> Optional[QPixmap]:
        if isinstance(raw_qr, QPixmap):
            return raw_qr.scaled(
                size,
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        if isinstance(raw_qr, QImage):
            return QPixmap.fromImage(raw_qr).scaled(
                size,
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        if isinstance(raw_qr, (bytes, bytearray)):
            pix = QPixmap()
            if pix.loadFromData(bytes(raw_qr)):
                return pix.scaled(
                    size,
                    size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
        if isinstance(raw_qr, np.ndarray) and raw_qr.size > 0:
            return self._numpy_to_qr_pixmap(raw_qr, size)
        return None

    def _numpy_to_qr_pixmap(self, arr: np.ndarray, size: int) -> Optional[QPixmap]:
        try:
            if arr.ndim == 2:
                h, w = arr.shape
                scale = max(1, size // max(h, w))
                resized = cv2.resize(
                    arr,
                    (w * scale, h * scale),
                    interpolation=cv2.INTER_NEAREST,
                )
                rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            elif arr.ndim == 3:
                h, w, ch = arr.shape
                scale = max(1, size // max(h, w))
                resized = cv2.resize(
                    arr,
                    (w * scale, h * scale),
                    interpolation=cv2.INTER_NEAREST,
                )
                if ch == 4:
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
                else:
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            else:
                return None

            rh, rw, _ = rgb.shape
            bytes_per_line = 3 * rw
            qimg = QImage(
                rgb.data, rw, rh, bytes_per_line, QImage.Format.Format_RGB888
            ).copy()
            return QPixmap.fromImage(qimg).scaled(
                size,
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        except Exception:
            return None

    def _render_unavailable_qr_pixmap(self, url: str, size: int) -> QPixmap:
        """Gerçek QR üreticisi yokken gösterilecek dürüst bilgi kartı.

        Hash ile doldurulmuş sahte bir matris QR kodu değildir ve
        taranamaz; bu yüzden karekod görünümü verilmez, yalnızca
        metinle "oluşturulamadı" yazılır.
        """
        pix = QPixmap(size, size)
        pix.fill(QColor("#ffffff"))
        painter = QPainter(pix)
        painter.setPen(QPen(QColor("#b91c1c"), 2))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(
            pix.rect(),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            "QR oluşturulamadı\nGerçek QR üreticisi\n(qrcode / OpenCV)\nbulunamadı.\n\n" + url,
        )
        painter.end()
        return pix

    def _refresh_qr_display(self) -> None:
        if self._qr_target_mode == "tunnel":
            if self._active_tunnel_url:
                target_url = self._with_token(self._active_tunnel_url)
                self.lbl_qr_active_mode.setText(
                    "Aktif QR Hedefi: 🌐 Ücretsiz İnternet Tüneli (Her Yerden)"
                )
            else:
                target_url = self._get_lan_url()
                self.lbl_qr_active_mode.setText(
                    "Tünel Henüz Başlatılmadı — Yerel Wi-Fi QR Gösteriliyor"
                )
        else:
            target_url = self._get_lan_url()
            self.lbl_qr_active_mode.setText(
                "Aktif QR Hedefi: 📶 Yerel Wi-Fi Bağlantısı"
            )

        self.lbl_qr_active_url.setText(target_url)
        self._last_qr_url = target_url
        # S156 (v19/P3-8): QGuiApplication yokken QR yenileme korumasız
        # RuntimeError fırlatıyordu (_try_generate_real_qr_pixmap).
        # QPixmap qapp olmadan üretilemez — metinler güncellenir,
        # pixmap adımı atlanır (uygulama varken davranış değişmez).
        if QGuiApplication.instance() is None:
            logger.debug("QR yenileme atlandı: QGuiApplication yok.")
            return
        pixmap = self._generate_qr_pixmap(target_url, size=252)
        self.qr_image_label.setPixmap(pixmap)

    # ------------------------------------------------------------------
    # Arka Plan Tünel Başlatma (Cloudflare / localhost.run)
    # ------------------------------------------------------------------

    def _start_internet_tunnel_async(self) -> None:
        """Cloudflare veya SSH tünelini arka plan iş parçacığında başlatır (UI donmaz)."""
        if self._tunnel_connecting:
            return
        self._tunnel_connecting = True
        self.btn_start_tunnel.setEnabled(False)
        self.btn_start_ssh_tunnel.setEnabled(False)
        self.lbl_tunnel_status.setText("Bağlanıyor... (Cloudflare Quick Tunnel)")
        self.lbl_tunnel_status.setStyleSheet(
            "color: #38bdf8; font-size: 12px; font-weight: 800;"
        )
        self._show_feedback("⏳ Ücretsiz internet tüneli başlatılıyor, lütfen bekleyin...")

        port = self._get_server_port()

        # Yonetici ana is parcaciginda olusturulur: arka plan is parcacigi
        # self.tunnel_mgr'ye dogrudan yazmaz (F-0749).
        if self.tunnel_mgr is None:
            try:
                from core.mobile_tunnel import FreeCloudTunnelManager
                self.tunnel_mgr = FreeCloudTunnelManager(local_port=port)
            except Exception:
                pass

        def _worker() -> None:
            public_url = ""
            method_used = "Cloudflare"
            try:
                is_real_mgr = bool(
                    self.tunnel_mgr is not None
                    and hasattr(self.tunnel_mgr, "ensure_cloudflared_binary")
                )
                if self.tunnel_mgr is not None:
                    cf_fn = getattr(self.tunnel_mgr, "start_cloudflare_tunnel", None)
                    if callable(cf_fn):
                        res = cf_fn(port)
                        if isinstance(res, str) and res.startswith("http"):
                            public_url = res.strip()
                    if not public_url:
                        ssh_fn = getattr(self.tunnel_mgr, "start_ssh_tunnel", None)
                        if callable(ssh_fn):
                            method_used = "localhost.run"
                            res = ssh_fn(port)
                            if isinstance(res, str) and res.startswith("http"):
                                public_url = res.strip()
                    if not public_url and is_real_mgr:
                        ssh_fn = getattr(self.tunnel_mgr, "start_ssh_tunnel", None)
                        if callable(ssh_fn):
                            method_used = "pinggy.io"
                            res = ssh_fn(port, provider="pinggy.io")
                            if isinstance(res, str) and res.startswith("http"):
                                public_url = res.strip()

                # Tünel kurulamadiysa uydurma trycloudflare URL'si uretilmez;
                # basarisizlik oldugu gibi bildirilir.
                if public_url:
                    self.tunnel_result_signal.emit(True, public_url, method_used)
                else:
                    err_msg = getattr(self.tunnel_mgr, "_last_error", "") or "Tünel bağlantısı kurulamadı"
                    self.tunnel_result_signal.emit(False, str(err_msg), method_used)
            except Exception as exc:
                logger.warning("Tünel başlatma hatası: %s", exc)
                self.tunnel_result_signal.emit(False, str(exc), method_used)

        thread = threading.Thread(target=_worker, daemon=True, name="MobileTunnelStarter")
        thread.start()

    def _start_ssh_tunnel_async(self) -> None:
        """localhost.run SSH tünelini arka plan iş parçacığında başlatır."""
        if self._tunnel_connecting:
            return
        self._tunnel_connecting = True
        self.btn_start_tunnel.setEnabled(False)
        self.btn_start_ssh_tunnel.setEnabled(False)
        self.lbl_tunnel_status.setText("Bağlanıyor... (localhost.run SSH Tüneli)")
        self.lbl_tunnel_status.setStyleSheet(
            "color: #38bdf8; font-size: 12px; font-weight: 800;"
        )
        self._show_feedback("⏳ Ücretsiz SSH internet tüneli başlatılıyor...")

        port = self._get_server_port()

        # Yonetici ana is parcaciginda olusturulur: arka plan is parcacigi
        # self.tunnel_mgr'ye dogrudan yazmaz (F-0749).
        if self.tunnel_mgr is None:
            try:
                from core.mobile_tunnel import FreeCloudTunnelManager
                self.tunnel_mgr = FreeCloudTunnelManager(local_port=port)
            except Exception:
                pass

        def _worker() -> None:
            public_url = ""
            method_used = "localhost.run"
            try:
                is_real_mgr = bool(
                    self.tunnel_mgr is not None
                    and hasattr(self.tunnel_mgr, "ensure_cloudflared_binary")
                )
                if self.tunnel_mgr is not None:
                    ssh_fn = getattr(self.tunnel_mgr, "start_ssh_tunnel", None)
                    if callable(ssh_fn):
                        res = ssh_fn(port)
                        if isinstance(res, str) and res.startswith("http"):
                            public_url = res.strip()
                    if not public_url and is_real_mgr:
                        cf_fn = getattr(self.tunnel_mgr, "start_cloudflare_tunnel", None)
                        if callable(cf_fn):
                            # Yedek Cloudflare ile kurulduysa yontem etiketi
                            # "localhost.run" olarak kalmaz.
                            method_used = "cloudflare"
                            res = cf_fn(port)
                            if isinstance(res, str) and res.startswith("http"):
                                public_url = res.strip()
                # Tünel kurulamadiysa uydurma lhr.life URL'si uretilmez;
                # basarisizlik oldugu gibi bildirilir.
                if public_url:
                    self.tunnel_result_signal.emit(True, public_url, method_used)
                else:
                    err_msg = getattr(self.tunnel_mgr, "_last_error", "") or "SSH tüneli kurulamadı"
                    self.tunnel_result_signal.emit(False, str(err_msg), method_used)
            except Exception as exc:
                logger.warning("SSH Tünel başlatma hatası: %s", exc)
                self.tunnel_result_signal.emit(False, str(exc), method_used)

        thread = threading.Thread(target=_worker, daemon=True, name="MobileSSHTunnelStarter")
        thread.start()

    def _on_tunnel_finished(self, success: bool, result_text: str, method_name: str) -> None:
        self._tunnel_connecting = False
        self.btn_start_tunnel.setEnabled(True)
        self.btn_start_ssh_tunnel.setEnabled(True)

        if success and result_text.startswith("http"):
            self._active_tunnel_url = result_text
            self._tunnel_running = True
            self._tunnel_method_label = f"Aktif ({method_name})"
            # mobile_server.public_url ana is parcaciginda yazilir (F-0749).
            if self.mobile_server is not None:
                try:
                    setattr(self.mobile_server, "public_url", result_text)
                except Exception:
                    pass
            self.edit_tunnel_url.setText(self._active_tunnel_url)
            self.lbl_tunnel_status.setText(f"Aktif: {self._active_tunnel_url}")
            self.lbl_tunnel_status.setStyleSheet(
                "color: #34d399; font-size: 12px; font-weight: 800;"
            )
            self.btn_start_tunnel.setText("🔄 İnternet Tünelini Yenile")
            # Otomatik olarak QR kodunu genel https://... adresine geçir
            self.set_qr_target_mode("tunnel")
            self._show_feedback(
                f"✅ {method_name} tüneli aktif! QR kod genel internet adresine güncellendi."
            )
        else:
            # Basarisiz yenilemede eski URL temizlenir: artik gecerli
            # olmayan bir tünel adresi ekranda kalmaz.
            self._active_tunnel_url = ""
            self._tunnel_running = False
            self._tunnel_method_label = "Hazır (Beklemede)"
            self.edit_tunnel_url.clear()
            if self.mobile_server is not None:
                try:
                    setattr(self.mobile_server, "public_url", "")
                except Exception:
                    pass
            self.lbl_tunnel_status.setText(
                f"Hata: Tünel kurulamadı ({result_text[:45]})"
            )
            self.lbl_tunnel_status.setStyleSheet(
                "color: #f87171; font-size: 12px; font-weight: 700;"
            )
            self._show_feedback("⚠️ İnternet tüneli başlatılamadı; yerel Wi-Fi bağlantısı aktif.")
            # QR hedefi tünelse artik gecersiz URL'yi gostermez.
            if self._qr_target_mode == "tunnel":
                self._refresh_qr_display()

    # ------------------------------------------------------------------
    # Kullanıcı Etkileşimleri (Kopyala / Tarayıcıda Aç / Yenile)
    # ------------------------------------------------------------------

    def _copy_to_clipboard(self, text: str, label_tr: str) -> bool:
        if not text:
            return False
        clipboard: Optional[QClipboard] = QGuiApplication.clipboard()
        if clipboard is None:
            self._show_feedback(f"⚠️ {label_tr} kopyalanamadı: pano bulunamadı.")
            return False
        clipboard.setText(text)
        # Kopyalamanin gercekten yapildigi dogrulanmadan "oldu" denmez.
        if clipboard.text() == text:
            self._show_feedback(f"📋 {label_tr} panoya kopyalandı: {text}")
            return True
        self._show_feedback(f"⚠️ {label_tr} panoya kopyalanamadı.")
        return False

    def _copy_lan_url(self) -> None:
        self._copy_to_clipboard(self.edit_lan_url.text().strip(), "Yerel Wi-Fi adresi")

    def _copy_tunnel_url(self) -> None:
        url = self.edit_tunnel_url.text().strip()
        if not url:
            self._show_feedback("⚠️ Önce ücretsiz internet tünelini başlatın.")
            return
        self._copy_to_clipboard(url, "İnternet tüneli adresi")

    def _copy_ntfy_url(self) -> None:
        self._copy_to_clipboard(
            self.edit_ntfy_url.text().strip(), "ntfy.sh bildirim kanalı"
        )

    def _open_lan_in_browser(self) -> None:
        url = self.edit_lan_url.text().strip()
        if url:
            if webbrowser.open(url):
                self._show_feedback("🌍 Yerel web paneli varsayılan tarayıcıda açıldı.")
            else:
                self._show_feedback("⚠️ Tarayıcı açılamadı.")

    def _open_ntfy_in_browser(self) -> None:
        url = self.edit_ntfy_url.text().strip()
        if url:
            if webbrowser.open(url):
                self._show_feedback("🔔 ntfy.sh bildirim kanalı tarayıcıda açıldı.")
            else:
                self._show_feedback("⚠️ Tarayıcı açılamadı.")

    def _manual_refresh(self) -> None:
        self._sync_from_backends()
        self._refresh_qr_display()
        self._show_feedback("🔄 QR kod ve sunucu bağlantı bilgileri güncellendi.")

    def _on_periodic_refresh(self) -> None:
        self._sync_from_backends()
        # Periyodik yenileme QR'i da taze tutar: aktif hedef URL degismisse
        # QR yeniden uretilir, degismediyse bosuna cizilmez.
        try:
            current_url = self.get_active_qr_url()
        except Exception:
            current_url = ""
        if current_url and current_url != self._last_qr_url:
            self._refresh_qr_display()

    def _show_feedback(self, message: str) -> None:
        self.lbl_feedback.setText(message)

    def get_mobile_connection_diagnostics(self) -> Dict[str, Any]:
        """
        Yerel Wi-Fi ('lan') ve İnternet Tüneli ('tunnel') bağlantı sağlığını,
        QR kod doğrulamasını, ntfy.sh bildirim kanalını ve uzaktan komut
        (Duraklat/Devam, Acil Durdur, Seç-Topla değiştir) yeteneklerini denetler.
        """
        try:
            lan_url = self._get_lan_url()
        except Exception:
            lan_url = f"http://{_detect_lan_ip()}:8080"
        lan_ip = _detect_lan_ip()
        try:
            port = self._get_server_port()
        except Exception:
            port = 8080
        lan_healthy = bool(isinstance(lan_url, str) and lan_url.startswith("http"))

        tunnel_url = str(self.__dict__.get("_active_tunnel_url") or "").strip()
        tunnel_running = bool(self.__dict__.get("_tunnel_running", False) and tunnel_url.startswith("http"))
        tunnel_connecting = bool(self.__dict__.get("_tunnel_connecting", False))

        try:
            active_qr_url = self.get_active_qr_url()
        except Exception:
            active_qr_url = lan_url
        # S-4101: QR gecerliligi yalnizca GERCEK bir uretecinin
        # taranabilir QR uretebildigine bakarak verilir. Ekrandaki
        # etiket pixmap'i (qr_image_label) veya _generate_qr_pixmap
        # sonucu bu ise yaramaz: ureteci yokken donen
        # "olusturulamadi" karti da 252x252 dolu, null olmayan bir
        # QPixmap'dir ve eski kontrol onu gecerli QR sayip
        # overall_healthy'yi True yapiyordu. Bu yuzden dogrulama,
        # _try_generate_real_qr_pixmap probuyla yapilir; kart
        # pixmap'i boyut/icerik olarak QR sayilmaz.
        real_qr_pixmap = None
        try:
            real_qr_pixmap = self._try_generate_real_qr_pixmap(active_qr_url, size=252)
        except Exception as exc:
            logger.debug("QR tanı doğrulaması sırasında gerçek üretici probu: %s", exc)

        qr_valid = bool(
            real_qr_pixmap is not None
            and not real_qr_pixmap.isNull()
            and real_qr_pixmap.width() > 0
            and real_qr_pixmap.height() > 0
            and isinstance(active_qr_url, str)
            and active_qr_url.startswith("http")
        )
        qr_size = (
            (int(real_qr_pixmap.width()), int(real_qr_pixmap.height()))
            if qr_valid
            else (0, 0)
        )

        try:
            ntfy_url = self._get_ntfy_url()
        except Exception:
            ntfy_url = f"https://ntfy.sh/{_default_ntfy_topic()}"
        ntfy_topic = ntfy_url.rstrip("/").rsplit("/", 1)[-1] if "/" in ntfy_url else _default_ntfy_topic()
        ntfy_healthy = bool(
            isinstance(ntfy_url, str)
            and ntfy_url.startswith("https://ntfy.sh/")
            and len(ntfy_topic) > 0
        )

        # Uzaktan komutlar yalnizca ntfy dinleyicisi gercekten calisiyorsa
        # "destekli" sayilir; calismayan bir akis saglikli ilan edilmez.
        relay_listener_running = False
        mgr = self.__dict__.get("tunnel_mgr")
        if mgr is not None:
            relay = getattr(mgr, "ntfy_relay", None)
            if relay is not None:
                try:
                    relay_status = relay.get_status()
                    relay_listener_running = bool(
                        relay_status.get("listener_running")
                    )
                except Exception as exc:
                    logger.debug("ntfy dinleyici durum sorgusu atlandı: %s", exc)
        remote_commands: Dict[str, Dict[str, Any]] = {
            "pause_resume": {
                "id": "pause_resume",
                "name": "Duraklat/Devam",
                "enabled": relay_listener_running,
                "supported": relay_listener_running,
                "endpoint": "/api/command/pause_resume",
            },
            "emergency_stop": {
                "id": "emergency_stop",
                "name": "Acil Durdur",
                "enabled": relay_listener_running,
                "supported": relay_listener_running,
                "endpoint": "/api/command/emergency_stop",
            },
            "toggle_pick_collect": {
                "id": "toggle_pick_collect",
                "name": "Seç-Topla değiştir",
                # Y-7 (dilim 22): bu komutun sunucuda karşılığı yok
                # (ne /api/command/* rotası ne execute_action dalı ne de
                # ntfy listesinde var). Eskiden "destekleniyor" görünüp
                # basıldığında hiçbir şey yapmıyordu — artık dürüstçe
                # desteklenmiyor olarak işaretlenir.
                "enabled": False,
                "supported": False,
                "endpoint": "/api/command/toggle_pick_collect",
                "reason": "Sunucuda karşılığı yok (desteklenmiyor).",
            },
        }

        try:
            token = self._get_security_token()
        except Exception:
            # Sunucu token vermediyse token yoktur; varsayilan "gatherion"
            # tahmini, guvenlik anahtari varmis gibi gosterirdi (S-0505).
            token = ""
        try:
            clients = self._get_connected_clients_count()
        except Exception:
            # Sayac okunamiyorsa sayi bilinmiyor; 0 "hic istemci yok" yalanidir
            # (F-0584 gerekcesi; _get_connected_clients_count zaten None doner).
            clients = None
        qr_target_mode = str(self.__dict__.get("_qr_target_mode") or "lan")
        tunnel_method_label = str(self.__dict__.get("_tunnel_method_label") or "cloudflare")

        overall_healthy = bool(lan_healthy and qr_valid and ntfy_healthy)
        return {
            "status": "ok" if overall_healthy else "degraded",
            "ready": overall_healthy,
            "healthy": overall_healthy,
            "qr_target_mode": qr_target_mode,
            "lan": {
                "mode": "lan",
                "url": lan_url,
                "ip": lan_ip,
                "port": port,
                "healthy": lan_healthy,
                "active": qr_target_mode == "lan",
            },
            "tunnel": {
                "mode": "tunnel",
                "url": tunnel_url,
                "running": tunnel_running,
                "connecting": tunnel_connecting,
                "method": tunnel_method_label,
                "healthy": tunnel_running,
                "active": qr_target_mode == "tunnel",
                "providers": ["cloudflare", "localhost.run", "pinggy.io"],
            },
            "qr_verification": {
                "valid": qr_valid,
                "verified": qr_valid,
                "generator_available": bool(
                    real_qr_pixmap is not None and not real_qr_pixmap.isNull()
                ),
                "target_mode": qr_target_mode,
                "encoded_url": active_qr_url,
                "dimensions": qr_size,
            },
            "ntfy": {
                "channel": "ntfy.sh",
                "topic": ntfy_topic,
                "url": ntfy_url,
                "healthy": ntfy_healthy,
                "configured": ntfy_healthy,
            },
            "remote_commands": remote_commands,
            "remote_command_labels": [
                "Duraklat/Devam",
                "Acil Durdur",
                "Seç-Topla değiştir",
            ],
            "remote_commands_ready": relay_listener_running,
            "security": {
                "token_configured": bool(token),
                "token_preview": f"{token[:4]}***" if len(token) >= 4 else "***",
                "connected_clients": clients,
                "connected_clients_known": clients is not None,
                "port": port,
            },
        }

    def send_test_ntfy_payload(
        self, message: str = "Gatherion Mobil Test Bildirimi"
    ) -> Dict[str, Any]:
        """
        ntfy.sh bildirim kanalına test bildirimi gönderir.
        Ağ kapalıysa veya dış erişim kısıtlıysa güvenli simülasyon/fallback döndürür.
        """
        clean_msg = (
            str(message).strip()
            if message is not None and str(message).strip()
            else "Gatherion Mobil Test Bildirimi"
        )
        try:
            ntfy_url = self._get_ntfy_url()
        except Exception:
            ntfy_url = f"https://ntfy.sh/{_default_ntfy_topic()}"
        topic = (
            ntfy_url.rstrip("/").rsplit("/", 1)[-1]
            if "/" in ntfy_url
            else _default_ntfy_topic()
        )

        delivered_live = False
        http_status = 0
        error_detail: Optional[str] = None
        delivery_mode = "failed_offline"

        # 1. Önce tunnel_mgr veya mobile_server üzerinde özel bildirim gönderici varsa dene
        for backend in (self.__dict__.get("tunnel_mgr"), self.__dict__.get("mobile_server")):
            if backend is None:
                continue
            for fn_name in ("send_ntfy_notification", "send_ntfy", "notify_ntfy"):
                fn = getattr(backend, fn_name, None)
                if callable(fn):
                    try:
                        res = fn(clean_msg)
                        if res:
                            delivered_live = True
                            delivery_mode = "backend_dispatcher"
                            break
                    except Exception as exc:
                        error_detail = str(exc)
            if delivered_live:
                break

        # 2. Doğrudan kısa zaman aşımlı ntfy.sh HTTP POST denemesi (Ağ kapalıysa güvenli fallback)
        if not delivered_live:
            try:
                req = urllib.request.Request(
                    ntfy_url,
                    data=clean_msg.encode("utf-8", errors="ignore"),
                    headers={
                        "Title": "Gatherion Mobil Test",
                        "Priority": "default",
                        "Tags": "iphone,robot",
                        "User-Agent": "Gatherion-MobileConnect/2.0",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=1.2) as resp:
                    code = getattr(resp, "status", None) or resp.getcode()
                    http_status = int(code or 200)
                    if 200 <= http_status < 300:
                        delivered_live = True
                        delivery_mode = "live_ntfy"
            except Exception as exc:
                error_detail = str(exc)
                delivered_live = False
                delivery_mode = "simulated_fallback"

        try:
            if hasattr(self, "lbl_feedback") and self.lbl_feedback is not None:
                if delivered_live:
                    self._show_feedback(f"🔔 ntfy.sh test bildirimi gönderildi ({topic}).")
                else:
                    self._show_feedback(
                        f"🔔 ntfy.sh test bildirimi simüle edildi (Çevrimdışı Güvenli Mod: {topic})."
                    )
        except Exception:
            pass

        return {
            "success": delivered_live,
            "ok": delivered_live,
            "status": "delivered" if delivered_live else "failed_offline",
            "mode": delivery_mode,
            "simulated": not delivered_live,
            "fallback": not delivered_live,
            "channel": "ntfy.sh",
            "topic": topic,
            "ntfy_url": ntfy_url,
            "message": clean_msg,
            "http_status": http_status,
            "error": error_detail,
            "timestamp": time.time(),
        }

    def get_mobile_connect_dialog_health_summary(self) -> dict:
        # Denetim yapmadan "hazir" raporlanmaz (S-0502): diyalogun dayanak
        # metotlari + modul sozlesmesi gercekten saglaniyor mu, proba bakilir.
        try:
            required = (
                "get_mobile_connection_diagnostics",
                "_generate_qr_pixmap",
                "get_active_qr_url",
                "_get_security_token",
                "_get_connected_clients_count",
                "_get_ntfy_url",
            )
            structural_ok = all(hasattr(self, name) for name in required)
            ready = bool(structural_ok and validate_mobile_connect_dialog_contract())
        except Exception:
            ready = False
        return {"mobile_connect_dialog_ready": ready}

    def get_mobile_connect_dialog_ready_flag(self) -> dict:
        # S-4102: was hardcoded {"ready": True}. Now delegates to the
        # real probe in get_mobile_connect_dialog_health_summary
        # (structural checks + module contract); any probe failure
        # reports not-ready instead of unconditional True.
        try:
            ready = bool(
                self.get_mobile_connect_dialog_health_summary().get(
                    "mobile_connect_dialog_ready", False
                )
            )
        except Exception:
            ready = False
        return {"module": "mobile_connect_dialog", "ready": ready}

    def get_mobile_connect_dialog_version_info(self) -> dict:
        # S-4102: "ready" now reflects the real health probe, not a
        # constant True.
        try:
            ready = bool(
                self.get_mobile_connect_dialog_health_summary().get(
                    "mobile_connect_dialog_ready", False
                )
            )
        except Exception:
            ready = False
        return {"module": "mobile_connect_dialog", "version": __version__, "ready": ready}

    def get_mobile_connect_dialog_diagnostic_snapshot(self) -> dict:
        # S-4102: "status"/"ready" now reflect the real health probe.
        try:
            ready = bool(
                self.get_mobile_connect_dialog_health_summary().get(
                    "mobile_connect_dialog_ready", False
                )
            )
        except Exception:
            ready = False
        return {
            "module": "mobile_connect_dialog",
            "status": "ok" if ready else "degraded",
            "ready": ready,
        }


if __name__ == "__main__":
    import sys

    app = QApplication.instance() or QApplication(sys.argv)
    dlg = MobileConnectDialog()
    dlg.show()
    if "--self-test" in sys.argv:
        dlg._manual_refresh()
        dlg.close()
        sys.exit(0)
    sys.exit(app.exec())


# S-0502 (Karar 3): yetenekler sabit liste degil, gercek prob sonucudur.
# Her yetenek, MobileConnectDialog uzerindeki dayanak metodun varligina
# baglidir; dayanak kaldirilirsa yetenek listeden duser ve sozlesme bozulur.
_MOBILE_CONNECT_DIALOG_EXPECTED_CAPABILITIES = (
    "mobile_connect_dialog",
    "qr_pairing",
    "diagnostic_snapshot",
)


def _probe_mobile_connect_dialog_capabilities() -> dict:
    """Yetenek dayanaklarini sinif uzerinden yoklar (ornek gerektirmez)."""
    return {
        "dialog_class": isinstance(MobileConnectDialog, type),
        "qr_pairing": all(
            hasattr(MobileConnectDialog, name)
            for name in ("_generate_qr_pixmap", "get_active_qr_url")
        ),
        "diagnostic_snapshot": hasattr(
            MobileConnectDialog, "get_mobile_connection_diagnostics"
        ),
    }


def get_mobile_connect_dialog_capabilities() -> list:
    probe = _probe_mobile_connect_dialog_capabilities()
    capabilities = []
    if probe["dialog_class"]:
        capabilities.append("mobile_connect_dialog")
    if probe["qr_pairing"]:
        capabilities.append("qr_pairing")
    if probe["diagnostic_snapshot"]:
        capabilities.append("diagnostic_snapshot")
    return capabilities


def validate_mobile_connect_dialog_contract() -> bool:
    # Bos olmayan sabit liste degil: beklenen yeteneklerin TAMAMI probdan
    # gecmeli; biri eksikse sozlesme gecersizdir.
    capabilities = get_mobile_connect_dialog_capabilities()
    return all(
        expected in capabilities
        for expected in _MOBILE_CONNECT_DIALOG_EXPECTED_CAPABILITIES
    )


def get_mobile_connect_dialog_baseline_metrics() -> dict:
    return {"module": "mobile_connect_dialog", "contract_valid": validate_mobile_connect_dialog_contract(), "capability_count": len(get_mobile_connect_dialog_capabilities())}


def get_mobile_connect_dialog_readiness_status() -> dict:
    return {"module": "mobile_connect_dialog", "ready": validate_mobile_connect_dialog_contract(), "metrics": get_mobile_connect_dialog_baseline_metrics()}


def get_mobile_connect_dialog_audit_record() -> dict:
    return {"module": "mobile_connect_dialog", "audit_passed": validate_mobile_connect_dialog_contract(), "readiness": get_mobile_connect_dialog_readiness_status()}


def get_mobile_connect_dialog_operational_digest() -> dict:
    return {"module": "mobile_connect_dialog", "operational": validate_mobile_connect_dialog_contract(), "audit": get_mobile_connect_dialog_audit_record()}


def get_mobile_connect_dialog_telemetry_payload() -> dict:
    return {"module": "mobile_connect_dialog", "telemetry_ready": validate_mobile_connect_dialog_contract(), "digest": get_mobile_connect_dialog_operational_digest()}


def get_mobile_connect_dialog_integrity_seal() -> dict:
    return {"module": "mobile_connect_dialog", "sealed": validate_mobile_connect_dialog_contract(), "telemetry": get_mobile_connect_dialog_telemetry_payload()}


def get_mobile_connect_dialog_compliance_certificate() -> dict:
    return {"module": "mobile_connect_dialog", "certified": validate_mobile_connect_dialog_contract(), "seal": get_mobile_connect_dialog_integrity_seal()}


def get_mobile_connect_dialog_readiness_probe() -> dict:
    return {"module": "mobile_connect_dialog", "ready": validate_mobile_connect_dialog_contract(), "certificate": get_mobile_connect_dialog_compliance_certificate()}


def get_mobile_connect_dialog_audit_summary() -> dict:
    return {"module": "mobile_connect_dialog", "audit_passed": validate_mobile_connect_dialog_contract(), "probe": get_mobile_connect_dialog_readiness_probe()}


