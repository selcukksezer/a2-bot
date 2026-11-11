# -*- coding: utf-8 -*-
"""Proje kökü yardımcısı (P3-6, K-19/K-20).

Repo içinde proje kökünü çözen tek paylaşılan yardımcıdır; daha önce her
modül kendi ``Path(__file__).resolve().parent.parent`` /
``dirname(dirname(abspath(__file__)))`` kopyasını tutuyordu.

Bu modül yalnızca stdlib'e dayanır (repo içi import yok, döngüsel import
riski yok); ``core``, ``ui`` ve ``tools`` altından güvenle import edilebilir.

Kural: kalıcı veri yolları CWD'ye göre değil proje köküne göre çözülür.
Mutlak yollar aynen geçer; göreli yollar köke göre çözülür.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(*parts: str) -> Path:
    """Proje köküne göre mutlak yol kurar: ``project_path("routes", "Altgard")``."""
    return PROJECT_ROOT.joinpath(*parts)


def resolve_project_path(path: Union[str, Path]) -> Path:
    """Mutlak yolu aynen geçirir, göreli yolu proje köküne göre çözer.

    CWD'ye göre çözmez: bot farklı bir çalışma dizininden başlatılsa da
    aynı dosyayı bulur.
    """
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def resolve_project_str(path: Union[str, Path]) -> str:
    """``resolve_project_path``'in str dönen sürümü (os.path tabanlı kod için)."""
    return os.fspath(resolve_project_path(path))
