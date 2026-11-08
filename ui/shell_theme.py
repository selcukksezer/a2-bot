# -*- coding: utf-8 -*-
"""Shared Gatherion shell look: prototype dark palette, amber accent, clean chrome."""

from __future__ import annotations

from typing import Any

CONFIDENCE_TIER_COLORS = {
    "confident": "#22c55e",
    "suspicious": "#fb923c",
    "low": "#ef4444",
}

CONFIDENCE_TIER_PALETTE = CONFIDENCE_TIER_COLORS


def get_confidence_badge_qss(confidence: float, trusted: bool = True) -> str:
    """Returns 3-tier badge QSS with crisp white text for >=85% green, 70-84% orange, <70% red."""
    c = float(confidence or 0.0)
    if c > 1.0:
        c = c / 100.0
    if c >= 0.85 and trusted:
        bg = CONFIDENCE_TIER_COLORS["confident"]
    elif c >= 0.70:
        bg = CONFIDENCE_TIER_COLORS["suspicious"]
    else:
        bg = CONFIDENCE_TIER_COLORS["low"]
    return f"background-color: {bg}; color: #ffffff; border-radius: 4px; padding: 2px 6px; font-weight: 700;"


SHELL_COLORS = {
    "bg": "#0a0c10",
    "panel": "#11151c",
    "panel_deep": "#0d1015",
    "line": "#1e2530",
    "text": "#eef1f5",
    "muted": "#9aa3af",
    "faint": "#6b7280",
    "kicker": "#6b7280",
    "accent": "#f5b50a",
    "accent_press": "#d89e06",
    "ok": "#34d399",
    "warn": "#fb923c",
    "danger": "#f87171",
    "sky": "#56c8f5",
    "purple": "#a78bfa",
}

SHELL_PAGE_FRAME_QSS = f"""
    QFrame#ShellPageFrame {{
        background-color: {SHELL_COLORS["panel"]};
        border: 1px solid {SHELL_COLORS["line"]};
        border-radius: 10px;
    }}
    QFrame#ShellPageFrameSeamless {{
        background-color: transparent;
        border: none;
        border-radius: 0px;
    }}
    QLabel#ShellPageKicker {{
        color: {SHELL_COLORS["kicker"]};
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 2.0px;
        padding: 0px;
        font-family: "Segoe UI", system-ui, sans-serif;
    }}
    QLabel#ShellPageTitle {{
        color: {SHELL_COLORS["text"]};
        font-size: 21px;
        font-weight: 800;
        padding: 0px;
        font-family: "Segoe UI", system-ui, sans-serif;
    }}
    QLabel#ShellPageSubtitle {{
        color: {SHELL_COLORS["muted"]};
        font-size: 12px;
        font-weight: 500;
        padding: 0px;
        font-family: "Segoe UI", system-ui, sans-serif;
    }}
    QFrame#ShellMetricChip {{
        background-color: {SHELL_COLORS["panel_deep"]};
        border: 1px solid {SHELL_COLORS["line"]};
        border-radius: 8px;
    }}
    QLabel#ShellMetricCaption {{
        color: {SHELL_COLORS["muted"]};
        font-size: 11px;
        font-weight: 600;
    }}
    QLabel#ShellMetricValue {{
        color: {SHELL_COLORS["text"]};
        font-size: 18px;
        font-weight: 700;
        font-family: "Segoe UI", system-ui, sans-serif;
    }}
    QPushButton#ShellPrimaryBtn {{
        background-color: {SHELL_COLORS["accent"]};
        color: #120d00;
        border: 1px solid {SHELL_COLORS["accent"]};
        border-radius: 7px;
        padding: 8px 14px;
        font-size: 12px;
        font-weight: 800;
        font-family: "Segoe UI", system-ui, sans-serif;
    }}
    QPushButton#ShellPrimaryBtn:hover {{
        background-color: {SHELL_COLORS["accent_press"]};
        color: #120d00;
    }}
    QPushButton#ShellPrimaryBtn:disabled {{
        background-color: #252b36;
        border-color: #252b36;
        color: #6b7280;
    }}
    QPushButton#ShellSecondaryBtn {{
        background-color: {SHELL_COLORS["panel"]};
        color: {SHELL_COLORS["text"]};
        border: 1px solid {SHELL_COLORS["line"]};
        border-radius: 7px;
        padding: 8px 14px;
        font-size: 12px;
        font-weight: 600;
    }}
    QPushButton#ShellSecondaryBtn:hover {{
        background-color: #171d26;
        border-color: #2c3545;
        color: #ffffff;
    }}
    QPushButton#ShellGhostBtn {{
        background-color: {SHELL_COLORS["panel_deep"]};
        color: {SHELL_COLORS["text"]};
        border: 1px solid {SHELL_COLORS["line"]};
        border-radius: 7px;
        padding: 7px 12px;
        font-size: 12px;
        font-weight: 600;
    }}
    QPushButton#ShellGhostBtn:hover {{
        color: #ffffff;
        background-color: #171d26;
        border-color: #2c3545;
    }}
    QProgressBar#ShellProgress {{
        background-color: {SHELL_COLORS["panel_deep"]};
        border: 1px solid {SHELL_COLORS["line"]};
        border-radius: 5px;
        text-align: center;
        color: #ffffff;
        font-size: 12px;
        font-weight: 700;
        min-height: 20px;
        max-height: 20px;
    }}
    QProgressBar#ShellProgress::chunk {{
        background-color: {SHELL_COLORS["ok"]};
        border-radius: 4px;
    }}
    QComboBox,
    QComboBox#ShellFilter,
    QComboBox#PipeCombo,
    QComboBox#PipeClassFilter,
    QComboBox#PipeClassCombo,
    QComboBox#ShellRegionCombo,
    QComboBox#ShellRouteCombo {{
        background-color: #111622;
        color: #f0f4f8;
        border: 1px solid {SHELL_COLORS["line"]};
        border-radius: 6px;
        padding: 5px 10px;
        font-size: 12px;
        min-height: 26px;
    }}
    QComboBox:hover,
    QComboBox#ShellFilter:hover,
    QComboBox#PipeCombo:hover {{
        border-color: #38bdf8;
    }}
    QComboBox::drop-down,
    QComboBox#ShellFilter::drop-down,
    QComboBox#PipeCombo::drop-down {{
        border: none;
        width: 22px;
    }}
    QComboBox QAbstractItemView,
    QComboBox QListView,
    QComboBox#ShellFilter QAbstractItemView,
    QComboBox#PipeCombo QAbstractItemView,
    QComboBox#PipeClassFilter QAbstractItemView,
    QComboBox#PipeClassCombo QAbstractItemView,
    QComboBox#ShellRegionCombo QAbstractItemView,
    QComboBox#ShellRouteCombo QAbstractItemView {{
        background-color: #111622;
        color: #ffffff !important;
        border: 1px solid #1e2530;
        selection-background-color: #2563eb;
        selection-color: #ffffff !important;
        font-size: 12px;
        padding: 4px;
        outline: none;
    }}
    QComboBox QAbstractItemView::item,
    QComboBox QListView::item {{
        color: #ffffff !important;
        background-color: #111622;
        min-height: 24px;
        padding: 4px 8px;
    }}
    QComboBox QAbstractItemView::item:selected,
    QComboBox QAbstractItemView::item:hover,
    QComboBox QListView::item:selected,
    QComboBox QListView::item:hover {{
        background-color: #2563eb;
        color: #ffffff !important;
    }}
    QScrollArea#ShellScroll {{
        background: transparent;
        border: none;
    }}
"""

SHELL_MAIN_QSS = f"""
    QMainWindow, QWidget#ShellCentral {{
        background-color: {SHELL_COLORS["bg"]};
        color: {SHELL_COLORS["text"]};
        font-family: "Segoe UI", system-ui, sans-serif;
        font-size: 13px;
    }}
    QFrame#ShellTopBar {{
        background-color: {SHELL_COLORS["panel_deep"]};
        border: none;
        border-bottom: 1px solid {SHELL_COLORS["line"]};
    }}
    QLabel#TopBrand {{
        color: {SHELL_COLORS["accent"]};
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.8px;
        font-family: "Segoe UI", system-ui, sans-serif;
    }}
    QLabel#TopPageMeta {{
        color: {SHELL_COLORS["muted"]};
        font-size: 12px;
        font-weight: 500;
        font-family: "Segoe UI", system-ui, sans-serif;
    }}
    QFrame#ShellNavRail {{
        background-color: {SHELL_COLORS["panel_deep"]};
        border-right: 1px solid {SHELL_COLORS["line"]};
    }}
    QLabel#RailBrand {{
        color: {SHELL_COLORS["accent"]};
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.8px;
        padding: 8px 10px;
        font-family: "Segoe UI", system-ui, sans-serif;
    }}
    QPushButton[shellNav="true"] {{
        text-align: left;
        background: transparent;
        color: {SHELL_COLORS["muted"]};
        border: none;
        border-left: 2px solid transparent;
        border-radius: 7px;
        padding: 8px 10px;
        font-size: 13px;
        font-weight: 500;
    }}
    QPushButton[shellNav="true"]:hover {{
        background: rgba(255, 255, 255, 0.03);
        color: {SHELL_COLORS["text"]};
    }}
    QPushButton[shellNav="true"]:checked {{
        background: {SHELL_COLORS["panel"]};
        color: #ffffff;
        border-left: 2px solid {SHELL_COLORS["accent"]};
        font-weight: 600;
    }}
    QPushButton#Rail_settings {{
        text-align: left;
        background: transparent;
        color: {SHELL_COLORS["muted"]};
        border: none;
        border-left: 2px solid transparent;
        border-radius: 6px;
        padding: 6px 8px;
        font-size: 12px;
        font-weight: 500;
    }}
    QPushButton#Rail_settings:hover {{
        background: rgba(255, 255, 255, 0.04);
        color: {SHELL_COLORS["text"]};
    }}
    QPushButton#Rail_settings:checked {{
        background: {SHELL_COLORS["panel"]};
        color: {SHELL_COLORS["accent"]};
        border-left: 2px solid {SHELL_COLORS["accent"]};
        font-weight: 700;
    }}
    QLabel#RailVersion {{
        color: {SHELL_COLORS["faint"]};
        font-family: Consolas, "Cascadia Mono", monospace;
        font-size: 10px;
    }}
    QFrame#ShellLogCard {{
        background-color: #05070a;
        border: 1px solid {SHELL_COLORS["line"]};
        border-radius: 10px;
    }}
    QFrame#ShellLogHead {{
        background-color: {SHELL_COLORS["panel_deep"]};
        border: none;
        border-bottom: 1px solid {SHELL_COLORS["line"]};
        border-top-left-radius: 9px;
        border-top-right-radius: 9px;
    }}
    QLabel#ShellLogTitle {{
        color: {SHELL_COLORS["muted"]};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.6px;
    }}
    QPushButton[logPill="true"] {{
        background: transparent;
        border: 1px solid {SHELL_COLORS["line"]};
        color: {SHELL_COLORS["muted"]};
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 600;
    }}
    QPushButton[logPill="true"]:hover {{
        color: {SHELL_COLORS["text"]};
        border-color: #2c3545;
    }}
    QPushButton[logPill="true"]:checked {{
        color: {SHELL_COLORS["accent"]};
        border-color: rgba(245, 181, 10, 0.45);
        background: rgba(245, 181, 10, 0.08);
    }}
    QPushButton#ShellLogClearBtn {{
        background: transparent;
        border: 1px solid {SHELL_COLORS["line"]};
        color: {SHELL_COLORS["danger"]};
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 600;
    }}
    QPushButton#ShellLogClearBtn:hover {{
        background: rgba(248, 113, 113, 0.1);
        border-color: rgba(248, 113, 113, 0.4);
    }}
    QPlainTextEdit#ShellLogDock {{
        background-color: #05070a;
        color: {SHELL_COLORS["muted"]};
        border: none;
        border-bottom-left-radius: 9px;
        border-bottom-right-radius: 9px;
        font-family: Consolas, "Cascadia Mono", monospace;
        font-size: 11px;
        padding: 6px 12px;
    }}
    QWidget#HomePage, QScrollArea#HomeScroll, QWidget#HomeScrollContent {{
        background-color: {SHELL_COLORS["bg"]};
        border: none;
    }}
    QFrame#HomeStatusCard {{
        background-color: {SHELL_COLORS["panel"]};
        border: 1px solid {SHELL_COLORS["line"]};
        border-radius: 10px;
    }}
    QLabel#HomeState {{
        color: {SHELL_COLORS["text"]};
        font-size: 20px;
        font-weight: 800;
        font-family: "Segoe UI", system-ui, sans-serif;
    }}
    QLabel#HomeRouteSummary {{
        color: {SHELL_COLORS["muted"]};
        font-size: 13px;
        font-weight: 500;
    }}
    QLabel#HomeRegion, QLabel#HomeRoute {{
        color: {SHELL_COLORS["text"]};
        font-size: 13px;
        font-weight: 600;
    }}
    QLabel#HomeHotkeyLine {{
        color: {SHELL_COLORS["faint"]};
        font-size: 11px;
        font-family: Consolas, "Cascadia Mono", monospace;
    }}
    QFrame#HomeStatsStrip {{
        background-color: {SHELL_COLORS["panel"]};
        border: 1px solid {SHELL_COLORS["line"]};
        border-radius: 10px;
    }}
    QFrame#HomeStat_gathered, QFrame#HomeStat_active,
    QFrame#HomeStat_failures, QFrame#HomeStat_yz, QFrame#HomeStat_session {{
        background-color: transparent;
        border: none;
        border-right: 1px solid {SHELL_COLORS["line"]};
        border-radius: 0px;
    }}
    QFrame#HomeStat_session {{
        border-right: none;
    }}
    QLabel#HomeStatCaption {{
        color: {SHELL_COLORS["faint"]};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.6px;
    }}
    QLabel#HomeStatValue {{
        color: {SHELL_COLORS["text"]};
        font-size: 22px;
        font-weight: 800;
        font-family: "Segoe UI", system-ui, sans-serif;
    }}
    QLabel#HomeStatSub {{
        color: {SHELL_COLORS["muted"]};
        font-size: 12px;
        font-weight: 500;
    }}
    QPushButton[homePrimaryBtn="true"] {{
        background: {SHELL_COLORS["accent"]};
        color: #120d00;
        border: 1px solid {SHELL_COLORS["accent"]};
        border-radius: 7px;
        padding: 8px 14px;
        font-size: 12px;
        font-weight: 800;
    }}
    QPushButton[homePrimaryBtn="true"]:hover {{
        background: {SHELL_COLORS["accent_press"]};
        border-color: {SHELL_COLORS["accent_press"]};
        color: #120d00;
    }}
    QPushButton[homeSecondaryBtn="true"] {{
        background: {SHELL_COLORS["panel_deep"]};
        color: {SHELL_COLORS["text"]};
        border: 1px solid {SHELL_COLORS["line"]};
        border-radius: 7px;
        padding: 8px 14px;
        font-size: 12px;
        font-weight: 600;
    }}
    QPushButton[homeSecondaryBtn="true"]:hover {{
        background: #171d26;
        border-color: #2c3545;
        color: #ffffff;
    }}
    QPushButton[homeShortcut="true"] {{
        background: {SHELL_COLORS["accent"]};
        color: #120d00;
        border: 1px solid {SHELL_COLORS["accent"]};
        border-radius: 7px;
        padding: 7px 12px;
        font-size: 12px;
        font-weight: 800;
    }}
    QPushButton[homeShortcut="true"]:hover {{
        background: {SHELL_COLORS["accent_press"]};
        color: #120d00;
    }}
    QPushButton[homeAction="true"] {{
        background: {SHELL_COLORS["panel"]};
        color: {SHELL_COLORS["text"]};
        border: 1px solid {SHELL_COLORS["line"]};
        border-radius: 7px;
        padding: 7px 8px;
        font-size: 12px;
        font-weight: 600;
    }}
    QPushButton[homeAction="true"]:hover {{
        background: #171d26;
        border-color: rgba(255, 255, 255, 0.16);
        color: #ffffff;
    }}
    QPushButton[homeActionVariant="go"] {{
        color: {SHELL_COLORS["ok"]};
        border-color: rgba(52, 211, 153, 0.30);
    }}
    QPushButton[homeActionVariant="go"]:hover {{
        background: rgba(52, 211, 153, 0.10);
        border-color: rgba(52, 211, 153, 0.55);
        color: {SHELL_COLORS["ok"]};
    }}
    QPushButton[homeActionVariant="star"] {{
        color: {SHELL_COLORS["accent"]};
        border-color: rgba(245, 181, 10, 0.35);
    }}
    QPushButton[homeActionVariant="star"]:hover {{
        background: rgba(245, 181, 10, 0.10);
        border-color: rgba(245, 181, 10, 0.60);
        color: {SHELL_COLORS["accent"]};
    }}
    QPushButton[homeActionVariant="danger"] {{
        color: {SHELL_COLORS["danger"]};
        border-color: rgba(248, 113, 113, 0.28);
    }}
    QPushButton[homeActionVariant="danger"]:hover {{
        background: rgba(248, 113, 113, 0.10);
        border-color: rgba(248, 113, 113, 0.55);
        color: {SHELL_COLORS["danger"]};
    }}
    QFrame#HomeDetailCard {{
        background-color: {SHELL_COLORS["panel"]};
        border: 1px solid {SHELL_COLORS["line"]};
        border-radius: 10px;
    }}
    QLabel#HomeCardTitle {{
        color: {SHELL_COLORS["muted"]};
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.6px;
    }}
    QLabel#HomeCardMeta {{
        color: {SHELL_COLORS["faint"]};
        font-size: 11px;
        font-family: Consolas, "Cascadia Mono", monospace;
    }}
    QFrame#F9Overlay {{
        background-color: {SHELL_COLORS["panel_deep"]};
    }}
    QFrame#F9Overlay QScrollArea {{
        background-color: {SHELL_COLORS["panel_deep"]};
        border: none;
    }}
    QFrame#F9Overlay QScrollArea > QWidget {{
        background-color: {SHELL_COLORS["panel_deep"]};
    }}
    QComboBox,
    QComboBox#ShellFilter,
    QComboBox#PipeCombo,
    QComboBox#PipeClassFilter,
    QComboBox#PipeClassCombo,
    QComboBox#ShellRegionCombo,
    QComboBox#ShellRouteCombo {{
        background-color: #111622;
        color: #f0f4f8;
        border: 1px solid {SHELL_COLORS["line"]};
        border-radius: 6px;
        padding: 5px 10px;
        font-size: 12px;
        min-height: 26px;
    }}
    QComboBox:hover,
    QComboBox#ShellFilter:hover,
    QComboBox#PipeCombo:hover {{
        border-color: #38bdf8;
    }}
    QComboBox::drop-down,
    QComboBox#ShellFilter::drop-down,
    QComboBox#PipeCombo::drop-down {{
        border: none;
        width: 22px;
    }}
    QComboBox QAbstractItemView,
    QComboBox QListView,
    QComboBox#ShellFilter QAbstractItemView,
    QComboBox#PipeCombo QAbstractItemView,
    QComboBox#PipeClassFilter QAbstractItemView,
    QComboBox#PipeClassCombo QAbstractItemView,
    QComboBox#ShellRegionCombo QAbstractItemView,
    QComboBox#ShellRouteCombo QAbstractItemView {{
        background-color: #111622;
        color: #ffffff !important;
        border: 1px solid #1e2530;
        selection-background-color: #2563eb;
        selection-color: #ffffff !important;
        font-size: 12px;
        padding: 4px;
        outline: none;
    }}
    QComboBox QAbstractItemView::item,
    QComboBox QListView::item {{
        color: #ffffff !important;
        background-color: #111622;
        min-height: 24px;
        padding: 4px 8px;
    }}
    QComboBox QAbstractItemView::item:selected,
    QComboBox QAbstractItemView::item:hover,
    QComboBox QListView::item:selected,
    QComboBox QListView::item:hover {{
        background-color: #2563eb;
        color: #ffffff !important;
    }}
"""


def apply_combobox_dark_style(combo: Any) -> None:
    """
    Applies high-contrast dark theme with bright white text (#ffffff) on dark background (#111622)
    to a QComboBox and its dropdown list (QAbstractItemView / QListView / combo.view()).
    Sets both QSS and explicit QPalette roles so Windows Qt6 always renders white readable text.
    """
    if combo is None:
        return
    try:
        from PyQt6.QtGui import QColor, QPalette
        from PyQt6.QtWidgets import QComboBox

        if isinstance(combo, QComboBox):
            # 1. Palette on combo itself
            cpal = combo.palette()
            cpal.setColor(QPalette.ColorRole.WindowText, QColor("#ffffff"))
            cpal.setColor(QPalette.ColorRole.Text, QColor("#ffffff"))
            cpal.setColor(QPalette.ColorRole.ButtonText, QColor("#ffffff"))
            cpal.setColor(QPalette.ColorRole.Base, QColor("#111622"))
            cpal.setColor(QPalette.ColorRole.Window, QColor("#111622"))
            cpal.setColor(QPalette.ColorRole.Highlight, QColor("#2563eb"))
            cpal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
            combo.setPalette(cpal)

            # 2. Direct QSS on combo
            combo.setStyleSheet(
                "QComboBox {"
                " background-color: #111622;"
                " color: #ffffff;"
                " border: 1px solid #1e2530;"
                " border-radius: 6px;"
                " padding: 4px 8px;"
                " font-size: 12px;"
                " font-weight: 500;"
                " min-height: 24px;"
                "}"
                "QComboBox:hover { border-color: #38bdf8; }"
                "QComboBox::drop-down { border: none; width: 20px; }"
                "QComboBox QAbstractItemView, QComboBox QListView, QListView {"
                " background-color: #111622 !important;"
                " color: #ffffff !important;"
                " selection-background-color: #2563eb !important;"
                " selection-color: #ffffff !important;"
                " border: 1px solid #1e2530;"
                " padding: 4px;"
                " outline: none;"
                "}"
                "QComboBox QAbstractItemView::item, QComboBox QListView::item, QListView::item {"
                " color: #ffffff !important;"
                " background-color: #111622;"
                " min-height: 24px;"
                " padding: 4px 8px;"
                "}"
                "QComboBox QAbstractItemView::item:selected, QComboBox QAbstractItemView::item:hover,"
                "QComboBox QListView::item:selected, QComboBox QListView::item:hover,"
                "QListView::item:selected, QListView::item:hover {"
                " background-color: #2563eb !important;"
                " color: #ffffff !important;"
                "}"
            )

            # 3. View palette and styling
            view = combo.view()
            if view is not None:
                view.setAutoFillBackground(True)
                vpal = view.palette()
                vpal.setColor(QPalette.ColorRole.Text, QColor("#ffffff"))
                vpal.setColor(QPalette.ColorRole.WindowText, QColor("#ffffff"))
                vpal.setColor(QPalette.ColorRole.Base, QColor("#111622"))
                vpal.setColor(QPalette.ColorRole.Window, QColor("#111622"))
                vpal.setColor(QPalette.ColorRole.Highlight, QColor("#2563eb"))
                vpal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
                vpal.setColor(QPalette.ColorRole.AlternateBase, QColor("#161c28"))
                view.setPalette(vpal)
                view.setStyleSheet(
                    "QListView, QAbstractItemView {"
                    " color: #ffffff !important;"
                    " background-color: #111622 !important;"
                    " selection-color: #ffffff !important;"
                    " selection-background-color: #2563eb !important;"
                    " font-size: 12px;"
                    " border: 1px solid #1e2530;"
                    " outline: none;"
                    " padding: 4px;"
                    "}"
                    "QListView::item, QAbstractItemView::item {"
                    " color: #ffffff !important;"
                    " background-color: #111622;"
                    " min-height: 24px;"
                    " padding: 4px 8px;"
                    "}"
                    "QListView::item:selected, QListView::item:hover,"
                    "QAbstractItemView::item:selected, QAbstractItemView::item:hover {"
                    " background-color: #2563eb !important;"
                    " color: #ffffff !important;"
                    "}"
                )
    except Exception:
        pass


apply_combobox_white_text_style = apply_combobox_dark_style
style_combobox_dark_view = apply_combobox_dark_style
style_combobox_dark_popup = apply_combobox_dark_style


def install_combobox_dark_theme_hook() -> None:
    """
    Globally hooks QComboBox.showPopup so that whenever ANY QComboBox in the application
    opens its dropdown popup on Windows Qt6, its view (QAbstractItemView / QListView)
    has its stylesheet and palette set to crisp white text (#ffffff) on dark background (#111622 / #0f172a).
    """
    try:
        from PyQt6.QtWidgets import QComboBox

        orig_show_popup = getattr(QComboBox, "showPopup", None)
        if orig_show_popup is None or getattr(orig_show_popup, "_dark_hooked", False):
            return

        def _dark_show_popup(self: QComboBox) -> None:
            try:
                apply_combobox_dark_style(self)
            except Exception:
                pass
            orig_show_popup(self)

        _dark_show_popup._dark_hooked = True
        QComboBox.showPopup = _dark_show_popup
    except Exception:
        pass


def apply_high_contrast_input_style(widget: Any) -> None:
    """
    Guarantees high-contrast dark theme readability across all tabs for QComboBox,
    QLineEdit, and QTableWidget instances (or any parent container holding them).
    """
    if widget is None:
        return
    try:
        from PyQt6.QtGui import QColor, QPalette
        from PyQt6.QtWidgets import QComboBox, QLineEdit, QTableWidget, QWidget

        def _style_line_edit(le: QLineEdit) -> None:
            pal = le.palette()
            pal.setColor(QPalette.ColorRole.Base, QColor("#111622"))
            pal.setColor(QPalette.ColorRole.Window, QColor("#111622"))
            pal.setColor(QPalette.ColorRole.Text, QColor("#ffffff"))
            pal.setColor(QPalette.ColorRole.WindowText, QColor("#ffffff"))
            pal.setColor(QPalette.ColorRole.PlaceholderText, QColor("#9aa3af"))
            pal.setColor(QPalette.ColorRole.Highlight, QColor("#2563eb"))
            pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
            le.setPalette(pal)
            le.setStyleSheet(
                "QLineEdit {"
                " background-color: #111622;"
                " color: #ffffff;"
                " border: 1px solid #1e2530;"
                " border-radius: 6px;"
                " padding: 5px 10px;"
                " font-size: 12px;"
                " selection-background-color: #2563eb;"
                " selection-color: #ffffff;"
                "}"
                "QLineEdit:focus {"
                " border-color: #f5b50a;"
                "}"
                "QLineEdit:disabled {"
                " background-color: #0d1015;"
                " color: #6b7280;"
                "}"
            )

        def _style_table_widget(tbl: QTableWidget) -> None:
            pal = tbl.palette()
            pal.setColor(QPalette.ColorRole.Base, QColor("#11151c"))
            pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#161c28"))
            pal.setColor(QPalette.ColorRole.Window, QColor("#11151c"))
            pal.setColor(QPalette.ColorRole.Text, QColor("#ffffff"))
            pal.setColor(QPalette.ColorRole.WindowText, QColor("#ffffff"))
            pal.setColor(QPalette.ColorRole.Button, QColor("#0d1015"))
            pal.setColor(QPalette.ColorRole.ButtonText, QColor("#eef1f5"))
            pal.setColor(QPalette.ColorRole.Highlight, QColor("#2563eb"))
            pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
            tbl.setPalette(pal)
            if tbl.viewport() is not None:
                tbl.viewport().setPalette(pal)
            tbl.setStyleSheet(
                "QTableWidget, QTableView {"
                " background-color: #11151c;"
                " alternate-background-color: #161c28;"
                " color: #ffffff;"
                " gridline-color: #1e2530;"
                " border: 1px solid #1e2530;"
                " border-radius: 6px;"
                " selection-background-color: #2563eb;"
                " selection-color: #ffffff;"
                " font-size: 12px;"
                "}"
                "QTableWidget::item, QTableView::item {"
                " color: #ffffff;"
                " padding: 4px 8px;"
                "}"
                "QTableWidget::item:selected, QTableView::item:selected {"
                " background-color: #2563eb;"
                " color: #ffffff;"
                "}"
                "QHeaderView::section {"
                " background-color: #0d1015;"
                " color: #eef1f5;"
                " border: none;"
                " border-bottom: 1px solid #1e2530;"
                " border-right: 1px solid #1e2530;"
                " padding: 5px 8px;"
                " font-size: 11px;"
                " font-weight: 700;"
                "}"
                "QTableCornerButton::section {"
                " background-color: #0d1015;"
                " border: 1px solid #1e2530;"
                "}"
            )

        if isinstance(widget, QComboBox):
            apply_combobox_dark_style(widget)
        elif isinstance(widget, QLineEdit):
            _style_line_edit(widget)
        elif isinstance(widget, QTableWidget):
            _style_table_widget(widget)

        if isinstance(widget, QWidget):
            for child_combo in widget.findChildren(QComboBox):
                apply_combobox_dark_style(child_combo)
            for child_le in widget.findChildren(QLineEdit):
                _style_line_edit(child_le)
            for child_tbl in widget.findChildren(QTableWidget):
                _style_table_widget(child_tbl)
    except Exception:
        pass


# Automatically install global hook on import
install_combobox_dark_theme_hook()

__all__ = [
    "CONFIDENCE_TIER_COLORS",
    "SHELL_COLORS",
    "SHELL_PAGE_FRAME_QSS",
    "SHELL_MAIN_QSS",
    "apply_combobox_dark_style",
    "apply_combobox_white_text_style",
    "style_combobox_dark_view",
    "style_combobox_dark_popup",
    "install_combobox_dark_theme_hook",
    "apply_high_contrast_input_style",
]
