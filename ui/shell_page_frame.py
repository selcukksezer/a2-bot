# -*- coding: utf-8 -*-
"""Shared page chrome for GatherionShell rail tabs.

One title block per page. Content goes in the returned layout — pages must not
repeat the same title. Qt stylesheet only.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from PyQt6.QtWidgets import QFrame, QLabel, QStackedWidget, QVBoxLayout, QWidget

from ui.shell_theme import CONFIDENCE_TIER_COLORS, SHELL_PAGE_FRAME_QSS

# tab_key -> (kicker, title, subtitle)
_TAB_META: Dict[str, Tuple[str, str, str]] = {
    "radar": ("KEŞİF", "Radar", "Minimap ve rota üzerinde canlı haritalama."),
    "map": ("ROTA", "Harita", "Noktalar, rota üyeliği ve oyuncu konumu."),
    "yolo": ("YZ", "YOLO fabrikası", "Etiketle, zenginleştir, RTX ile eğit."),
    "calibration": ("GÖRÜNTÜ", "Kalibrasyon", "F/E, minimap oku ve harita şablonları."),
    "stats": ("ÖLÇÜM", "İstatistik", "Oturum verimi ve kaynak dağılımı."),
    "mobile": ("UZAKTAN", "Mobil", "QR panel ve bildirim kanalı."),
    "guide": ("YARDIM", "Kılavuz", "Kurulum ve sistem tanılama."),
    "logs": ("KAYIT", "Günlük", "bot.log akışı."),
    "settings": ("YAPILANDIRMA", "Ayarlar", "Toplama, mola, kısayol ve rota."),
}


def _is_shell_stack_embedded(page: QWidget) -> bool:
    depth = 0
    w: Optional[QWidget] = page
    while w is not None and depth < 24:
        if isinstance(w, QStackedWidget) and w.objectName() == "ShellStack":
            return True
        w = w.parentWidget()
        depth += 1
    return False


def wrap_shell_tab_root_layout(page: QWidget, tab_key: str) -> QVBoxLayout:
    """Root layout: title chrome once, then an inner host for page content."""
    kicker_text, title_text, subtitle_text = _TAB_META.get(tab_key, ("", "Sayfa", ""))
    try:
        setattr(page, "_shell_tab_key", tab_key)
        setattr(page, "_shell_status_badge", kicker_text or "AKTİF")
        page.setProperty("shellTabKey", tab_key)
        page.setProperty("shellStatusBadge", kicker_text or "AKTİF")
    except Exception:
        pass

    if not _is_shell_stack_embedded(page):
        # F-0689: a page used outside the shell stack used to lose its title
        # chrome silently. Render the same kicker/title/subtitle here.
        plain = QVBoxLayout(page)
        plain.setContentsMargins(14, 12, 14, 12)
        plain.setSpacing(0)
        suppress_chrome = tab_key in ("yolo", "settings", "stats")
        if not suppress_chrome:
            if kicker_text:
                kicker = QLabel(kicker_text, page)
                kicker.setObjectName("ShellPageKicker")
                plain.addWidget(kicker)
            title_lbl = QLabel(title_text, page)
            title_lbl.setObjectName("ShellPageTitle")
            plain.addWidget(title_lbl)
            if subtitle_text:
                sub_lbl = QLabel(subtitle_text, page)
                sub_lbl.setObjectName("ShellPageSubtitle")
                sub_lbl.setWordWrap(True)
                plain.addWidget(sub_lbl)
        inner_host = QWidget(page)
        inner = QVBoxLayout(inner_host)
        inner.setContentsMargins(0, 12, 0, 0)
        inner.setSpacing(10)
        plain.addWidget(inner_host, stretch=1)
        existing = page.styleSheet() or ""
        if "ShellPageFrame" not in existing:
            page.setStyleSheet((existing + "\n" + SHELL_PAGE_FRAME_QSS).strip())
        return inner

    # YOLO, Settings, and Stats pages own their prototype headers — skip duplicate shell chrome.
    suppress_chrome = tab_key in ("yolo", "settings", "stats")

    outer = QVBoxLayout(page)
    outer.setContentsMargins(
        0 if suppress_chrome else 14,
        0 if suppress_chrome else 12,
        0 if suppress_chrome else 14,
        0 if suppress_chrome else 12,
    )
    outer.setSpacing(0)

    frame = QFrame(page)
    frame.setObjectName("ShellPageFrameSeamless" if suppress_chrome else "ShellPageFrame")
    frame_l = QVBoxLayout(frame)
    frame_l.setContentsMargins(
        0 if suppress_chrome else 20,
        0 if suppress_chrome else 16,
        0 if suppress_chrome else 20,
        0 if suppress_chrome else 16,
    )
    frame_l.setSpacing(6)

    if not suppress_chrome:
        if kicker_text:
            kicker = QLabel(kicker_text, frame)
            kicker.setObjectName("ShellPageKicker")
            frame_l.addWidget(kicker)

        title_lbl = QLabel(title_text, frame)
        title_lbl.setObjectName("ShellPageTitle")
        frame_l.addWidget(title_lbl)

        if subtitle_text:
            sub_lbl = QLabel(subtitle_text, frame)
            sub_lbl.setObjectName("ShellPageSubtitle")
            sub_lbl.setWordWrap(True)
            frame_l.addWidget(sub_lbl)

    inner_host = QWidget(frame)
    inner = QVBoxLayout(inner_host)
    if tab_key in ("settings", "stats"):
        inner.setContentsMargins(12, 10, 14, 10)
        inner.setSpacing(10)
    else:
        inner.setContentsMargins(
            20 if suppress_chrome else 0,
            16 if suppress_chrome else 12,
            22 if suppress_chrome else 0,
            16 if suppress_chrome else 0,
        )
        inner.setSpacing(12 if suppress_chrome else 10)
    frame_l.addWidget(inner_host, stretch=1)

    outer.addWidget(frame, stretch=1)

    existing = page.styleSheet() or ""
    if "ShellPageFrame" not in existing:
        page.setStyleSheet((existing + "\n" + SHELL_PAGE_FRAME_QSS).strip())

    return inner


def shell_tab_breadcrumb(page: QWidget, tab_key: str) -> str:
    kicker, title, _ = _TAB_META.get(tab_key, ("", "Sayfa", ""))
    if kicker:
        return f"{title} · {kicker}"
    return title


def get_shell_page_metadata(page_widget: Any) -> Dict[str, Any]:
    """
    Queries the active tab key and status badge metadata for a shell tab page widget
    (or tab key string) wrapped by wrap_shell_tab_root_layout.
    """
    if isinstance(page_widget, str):
        tab_key = page_widget.strip()
        kicker, title, subtitle = _TAB_META.get(tab_key, ("", "Sayfa", ""))
        status_badge = kicker or "AKTİF"
        return {
            "tab_key": tab_key,
            "tab_id": tab_key,
            "kicker": kicker,
            "title": title,
            "subtitle": subtitle,
            "status_badge": status_badge,
            "badge": status_badge,
            "breadcrumb": shell_tab_breadcrumb(None, tab_key),  # type: ignore[arg-type]
            "suppress_chrome": tab_key in ("yolo", "settings", "stats"),
            "embedded": False,
        }

    tab_key = ""
    status_badge = ""
    embedded = False

    if isinstance(page_widget, QWidget):
        embedded = _is_shell_stack_embedded(page_widget)
        w: Optional[QWidget] = page_widget
        depth = 0
        while w is not None and depth < 12:
            tk = getattr(w, "_shell_tab_key", None) or w.property("shellTabKey")
            if isinstance(tk, str) and tk:
                tab_key = tk
                sb = getattr(w, "_shell_status_badge", None) or w.property("shellStatusBadge")
                if isinstance(sb, str) and sb:
                    status_badge = sb
                break
            w = w.parentWidget()
            depth += 1

        if not tab_key:
            title_lbl = page_widget.findChild(QLabel, "ShellPageTitle")
            kicker_lbl = page_widget.findChild(QLabel, "ShellPageKicker")
            title_val = title_lbl.text().strip() if title_lbl is not None else ""
            kicker_val = kicker_lbl.text().strip() if kicker_lbl is not None else ""
            for k, (meta_kicker, meta_title, _) in _TAB_META.items():
                if (title_val and meta_title == title_val) or (kicker_val and meta_kicker == kicker_val):
                    tab_key = k
                    break

        if not status_badge:
            kicker_lbl = page_widget.findChild(QLabel, "ShellPageKicker")
            if kicker_lbl is not None and kicker_lbl.text().strip():
                status_badge = kicker_lbl.text().strip()

    kicker, title, subtitle = _TAB_META.get(tab_key, ("", "Sayfa", ""))
    if not status_badge:
        status_badge = kicker or "AKTİF"

    return {
        "tab_key": tab_key,
        "tab_id": tab_key,
        "kicker": kicker,
        "title": title,
        "subtitle": subtitle,
        "status_badge": status_badge,
        "badge": status_badge,
        "breadcrumb": shell_tab_breadcrumb(page_widget if isinstance(page_widget, QWidget) else None, tab_key),  # type: ignore[arg-type]
        "suppress_chrome": tab_key in ("yolo", "settings", "stats"),
        "embedded": embedded,
    }


def set_page_status_badge(page_widget: Any, badge_text: str, tier: str = "confident") -> Dict[str, str]:
    """Updates the shell page status badge and returns the applied badge metadata."""
    text = str(badge_text or "AKTİF").strip()
    # P3-7: tier renkleri tek kaynaktan (ui.shell_theme.CONFIDENCE_TIER_COLORS).
    color_map = CONFIDENCE_TIER_COLORS
    color = color_map.get(str(tier or "confident").lower(), CONFIDENCE_TIER_COLORS["confident"])
    if isinstance(page_widget, QWidget):
        try:
            setattr(page_widget, "_shell_status_badge", text)
            page_widget.setProperty("shellStatusBadge", text)
            page_widget.setProperty("shellStatusBadgeColor", color)
            kicker_lbl = page_widget.findChild(QLabel, "ShellPageKicker")
            if kicker_lbl is not None:
                # F-0765: the tier colour was computed but never applied to
                # any widget. Colour the kicker label (only its colour is set,
                # so the QSS font rules still apply).
                kicker_lbl.setText(text)
                kicker_lbl.setStyleSheet(f"color: {color};")
        except Exception:
            pass
    return {"badge": text, "tier": tier, "color": color}


__all__ = [
    "_TAB_META",
    "wrap_shell_tab_root_layout",
    "shell_tab_breadcrumb",
    "get_shell_page_metadata",
    "set_page_status_badge",
]
