# -*- coding: utf-8 -*-
"""
Oyun İçi Ekonomi, Pazar (Trade Broker) Fiyat Tarayıcı, En Kârlı Rota Seçici,
Otomatik Üretim (Auto-Crafting / Substance Morphing) Kuyruğu ve Kinah Simülatörü.

Bölüm 2 Bileşenleri:
1. TradeBrokerPriceScanner:
   - Bilgisayarlı Görü (OpenCV) ile oyun içi Pazar (Trade Broker) penceresini tarar
     (`scan_broker_window(frame)`) ve tüm Aion 2 materyalleri için kalıcı fiyat
     veritabanını (`config/broker_prices.json`) yönetir.
   - `select_most_profitable_route(route_manager, region)`: mevcut rotaların düğüm
     materyal dağılımını, 3x hasat çarpanını (`max_harvests`) ve güncel birim
     fiyatları kullanarak saatlik beklenen Kinah/Altın getirisini (`expected_kinah_per_hour`)
     hesaplar ve en yüksek kârlı rotayı otomatik olarak önerir/aktifleştirir.
2. AutoCraftingMorphQueue:
   - Toplanan ham materyal envanterini takip eder (`record_gathered_material`).
   - Belirlenen eşik değerine (örn. 20 veya 50 adet) ulaşan ham materyaller için
     `process_crafting_queue(input_ctrl, test_mode)` ile Üretim / Dönüştürme
     (Crafting / Substance Morphing) penceresini açarak külçe, iksir, yemek veya
     dönüştürülmüş materyal üretimini gerçekleştirir.
3. SessionProfitSimulator:
   - Masaüstü HUD ve Mobil Gösterge Paneli için gerçek zamanlı finansal telemetri üretir
     (`get_session_economy_summary(elapsed_hours)`):
     `net_kinah_earned`, `kinah_per_hour`, `formatted_summary_tr`, `material_breakdown`.
"""

from __future__ import annotations

import copy
import inspect
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from version import __version__

from core.corridor_navigator import WORLD_MAP_METERS_PER_PX
from core.project_root import resolve_project_path

logger = logging.getLogger("EconomyBroker")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
logger.propagate = False


# ==============================================================================
# VARSAYILAN AION 2 MATERYAL VE ÜRÜN PAZAR FİYATLARI (KINAH / ALTIN)
# ==============================================================================
DEFAULT_BROKER_PRICES: Dict[str, int] = {
    # Madenler (Ores & Ingots)
    "Orichalcum Ore": 14500,
    "Fine Orichalcum Ore": 42000,
    "Pure Orichalcum Ore": 115000,
    "Adamantium Ore": 18200,
    "Platinum Ore": 24500,
    "Titanium Ore": 11800,
    "Iron Ore": 6400,
    # Bitkiler ve Otlar (Herbs & Plants)
    "Aria": 9800,
    "Fresh Aria": 28500,
    "Targena": 12400,
    "Azpha": 15600,
    "Calendula": 13200,
    "Mela": 11500,
    "Cippo": 8900,
    "Angelica": 9400,
    "Taro": 9100,
    "Dyam": 8700,
    "Rhea": 10800,
    "Ginseng": 13900,
    "Umbel": 11200,
    # Değerli Taşlar (Gemstones)
    "Sapphire Gemstone": 38000,
    "Ruby Gemstone": 36500,
    "Diamond Gemstone": 64000,
    "Aquamarine Gemstone": 27500,
    "Lazuli Gemstone": 25000,
    "Garnet Gemstone": 29500,
    "Moonstone Gemstone": 31000,
    "Topaz Gemstone": 34000,
    "Peridot Gemstone": 32500,
    "Emerald Gemstone": 44000,
    "Amethyst Gemstone": 33000,
    # Kereste ve Kütükler (Logs & Timber)
    "Yggdrasil Log": 19500,
    "Asvata Log": 16800,
    # Hava Kaynakları (Odyle & Aether)
    "Odyle": 31500,
    "Aether": 26000,
    "Burning Aether": 35000,
    "Pure Aether": 58000,
    # İşlenmiş / Üretilmiş Ürünler (Crafted & Morphed Outputs)
    "Orichalcum Ingot": 36000,
    "Fine Orichalcum Ingot": 98000,
    "Pure Orichalcum Ingot": 265000,
    "Adamantium Ingot": 45000,
    "Greater Healing Potion": 27000,
    "Fine Recovery Elixir": 72000,
    "Targena Wind Serum": 31000,
    "Azpha Courage Scroll": 39500,
    "Calendula Panacea": 33500,
    "Mela Fruit Tart": 29500,
    "Polished Sapphire": 92000,
    "Polished Ruby": 88000,
    "Brilliant Diamond": 158000,
    "Yggdrasil Composite Plank": 48500,
    "Asvata Reinforced Timber": 41500,
    "Refined Odyle Crystal": 79000,
    "Condensed Aether Crystal": 65000,
}

# Alias for compatibility with ActiveLearningLoop requirement
DEFAULT_AION2_MATERIAL_PRICES = DEFAULT_BROKER_PRICES

# P0-7 / K-12 — TEK FİYAT KAYNAĞI: Tüketilebilir eşya maliyeti için ayrı bir
# fiyat tablosu YOKTUR. Eskiden buradaki DEFAULT_CONSUMABLE_COSTS tablosu,
# broker tablosundaki (DEFAULT_BROKER_PRICES) 6 ürünle 8-10 kat çelişiyordu
# (örn. Greater Healing Potion 27000 broker / 3500 "maliyet") ve kalan
# 16 kaleminin hiçbir kaynağı yoktu (tarama/prob/test/yorumda geçmiyor).
# Doğru kaynak broker tablosudur: config/broker_prices.json canlı taramayla
# beslenir ve 6 ürünün değerini de aynen doğrular
# (27000/72000/31000/39500/33500/29500). Tüketilen bir eşyanın maliyeti,
# fırsat maliyeti olarak broker fiyatıdır; calculate_net_hourly_kinah bunu
# _known_unit_price ile (canlı tarayıcı fiyatları dahil) aynı yoldan çözer.
# Broker tablosunda bilinmeyen tüketilebilir uydurma fiyatla
# maliyetlendirilmez — 0 sayılır ve `unpriced_consumables` altında raporlanır
# (S-0602 doktrini). Eski 16 kalemin (Healing Potion 1200, Mana Potion 1400,
# "Potion" 1500, "İksir" 1500 vb.) maliyeti kaynakta doğrulanamadı — uydurma
# sayı korunmadı, "doğrulanamadı" olarak işaretlidir; gerçek fiyat
# broker taramasıyla tabloya girince maliyet otomatik doğru çıkar.

# Görüntü İşleme (CV) ile Pazar Satırı İkon Renk İmzaları (HSV aralıkları -> Materyal eşlemesi)
BROKER_ICON_HSV_SIGNATURES: List[Tuple[str, Tuple[int, int, int], Tuple[int, int, int]]] = [
    ("Diamond Gemstone", (88, 25, 210), (108, 120, 255)),
    ("Sapphire Gemstone", (105, 140, 130), (128, 255, 255)),
    ("Ruby Gemstone", (0, 150, 130), (10, 255, 255)),
    ("Odyle", (130, 110, 150), (160, 255, 255)),
    ("Pure Orichalcum Ore", (18, 170, 185), (30, 255, 255)),
    ("Fine Orichalcum Ore", (48, 130, 140), (72, 255, 245)),
    ("Orichalcum Ore", (14, 95, 115), (26, 210, 215)),
    ("Fresh Aria", (72, 135, 150), (88, 255, 255)),
    ("Aria", (35, 85, 115), (55, 225, 225)),
    ("Targena", (112, 80, 120), (130, 210, 235)),
    ("Azpha", (155, 95, 120), (175, 240, 240)),
    ("Calendula", (22, 145, 150), (35, 255, 250)),
    ("Mela", (5, 125, 125), (16, 245, 235)),
    ("Yggdrasil Log", (10, 75, 70), (22, 195, 155)),
    ("Asvata Log", (16, 60, 80), (28, 175, 165)),
]


# Imza araliklarinin numpy dizisi halleri: _classify_row_material her satirda
# 15 imza icin np.array(lower/upper) kuruyordu (tarama basina 240 tahsis).
# Degerler BROKER_ICON_HSV_SIGNATURES ile birebir ayni; yalnizca bir kez kurulur.
BROKER_ICON_HSV_SIGNATURE_ARRAYS: List[Tuple[str, np.ndarray, np.ndarray]] = [
    (name, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    for name, lower, upper in BROKER_ICON_HSV_SIGNATURES
]


def _pooled_std_bgr(bgr: np.ndarray) -> float:
    """
    BGR goruntunun tum kanallarinda havuzlanmis (pooled) standart sapmasi.

    ``float(np.std(bgr))`` ile matematiksel olarak ayni degerdir
    (kanal bazli E[X^2] - E[X]^2 birlestirmesi; olculen fark ~3e-14),
    ancak OpenCV'nin tek gecisli ``meanStdDev``'i numpy'nin iki gecisli
    ``_var``'indan ~2-4.5x hizlidir (1920x1080 ROI'de olculdu).
    OpenCV yolu basarisiz olursa np.std yedegine duser.
    """
    try:
        mean, std = cv2.meanStdDev(bgr)
        m = mean.ravel().astype(np.float64)
        s = std.ravel().astype(np.float64)
        var = float(np.mean(s * s + m * m) - float(np.mean(m)) ** 2)
        return math.sqrt(max(0.0, var))
    except Exception:
        return float(np.std(bgr))


def format_kinah_tr(amount: Union[int, float]) -> str:
    """
    Kinah (Altın) tutarını okunabilir Türkçe kısa formata dönüştürür.
    Örn: 1840000 -> '1.84M', 42500 -> '42.5K', 850 -> '850'
    """
    val = float(amount)
    sign = "-" if val < 0 else ""
    abs_val = abs(val)

    def _short(divisor: float, suffix: str) -> str:
        # '42.50K' degil, dokumandaki gibi '42.5K' uret: sondaki sifirlari at.
        text = f"{abs_val / divisor:.2f}".rstrip("0").rstrip(".")
        return f"{sign}{text}{suffix}"

    if abs_val >= 1_000_000_000.0:
        return _short(1_000_000_000.0, "B")
    if abs_val >= 1_000_000.0:
        return _short(1_000_000.0, "M")
    if abs_val >= 1_000.0:
        return _short(1_000.0, "K")
    return f"{sign}{int(round(abs_val))}"


@dataclass
class CraftingRecipe:
    """Ham materyalin otomatik üretim veya dönüştürme (morph) reçetesi."""

    raw_material: str
    output_item: str
    threshold: int = 20
    batch_input_count: int = 10
    output_quantity: int = 5
    category: str = "smelting"
    window_hotkey: str = "p"
    craft_duration_sec: float = 1.8
    tab_norm_coords: Tuple[float, float] = (0.34, 0.24)
    recipe_norm_coords: Tuple[float, float] = (0.38, 0.36)
    craft_all_norm_coords: Tuple[float, float] = (0.62, 0.74)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_material": self.raw_material,
            "output_item": self.output_item,
            "threshold": int(self.threshold),
            "batch_input_count": int(self.batch_input_count),
            "output_quantity": int(self.output_quantity),
            "category": self.category,
            "window_hotkey": self.window_hotkey,
            "craft_duration_sec": float(self.craft_duration_sec),
            "tab_norm_coords": list(self.tab_norm_coords),
            "recipe_norm_coords": list(self.recipe_norm_coords),
            "craft_all_norm_coords": list(self.craft_all_norm_coords),
        }


DEFAULT_CRAFTING_RECIPES: Dict[str, CraftingRecipe] = {
    "Orichalcum Ore": CraftingRecipe(
        raw_material="Orichalcum Ore",
        output_item="Orichalcum Ingot",
        threshold=20,
        batch_input_count=10,
        output_quantity=5,
        category="smelting",
        window_hotkey="p",
        craft_duration_sec=1.6,
        tab_norm_coords=(0.32, 0.23),
        recipe_norm_coords=(0.37, 0.34),
        craft_all_norm_coords=(0.62, 0.74),
    ),
    "Fine Orichalcum Ore": CraftingRecipe(
        raw_material="Fine Orichalcum Ore",
        output_item="Fine Orichalcum Ingot",
        threshold=20,
        batch_input_count=10,
        output_quantity=5,
        category="smelting",
        window_hotkey="p",
        craft_duration_sec=1.8,
        tab_norm_coords=(0.32, 0.23),
        recipe_norm_coords=(0.37, 0.38),
        craft_all_norm_coords=(0.62, 0.74),
    ),
    "Pure Orichalcum Ore": CraftingRecipe(
        raw_material="Pure Orichalcum Ore",
        output_item="Pure Orichalcum Ingot",
        threshold=20,
        batch_input_count=10,
        output_quantity=5,
        category="smelting",
        window_hotkey="p",
        craft_duration_sec=2.0,
        tab_norm_coords=(0.32, 0.23),
        recipe_norm_coords=(0.37, 0.42),
        craft_all_norm_coords=(0.62, 0.74),
    ),
    "Aria": CraftingRecipe(
        raw_material="Aria",
        output_item="Greater Healing Potion",
        threshold=20,
        batch_input_count=10,
        output_quantity=5,
        category="alchemy",
        window_hotkey="p",
        craft_duration_sec=1.5,
        tab_norm_coords=(0.38, 0.23),
        recipe_norm_coords=(0.37, 0.35),
        craft_all_norm_coords=(0.62, 0.74),
    ),
    "Fresh Aria": CraftingRecipe(
        raw_material="Fresh Aria",
        output_item="Fine Recovery Elixir",
        threshold=20,
        batch_input_count=10,
        output_quantity=5,
        category="alchemy",
        window_hotkey="p",
        craft_duration_sec=1.7,
        tab_norm_coords=(0.38, 0.23),
        recipe_norm_coords=(0.37, 0.39),
        craft_all_norm_coords=(0.62, 0.74),
    ),
    "Targena": CraftingRecipe(
        raw_material="Targena",
        output_item="Targena Wind Serum",
        threshold=20,
        batch_input_count=10,
        output_quantity=5,
        category="alchemy",
        window_hotkey="p",
        craft_duration_sec=1.6,
        tab_norm_coords=(0.38, 0.23),
        recipe_norm_coords=(0.37, 0.43),
        craft_all_norm_coords=(0.62, 0.74),
    ),
    "Azpha": CraftingRecipe(
        raw_material="Azpha",
        output_item="Azpha Courage Scroll",
        threshold=20,
        batch_input_count=10,
        output_quantity=5,
        category="alchemy",
        window_hotkey="p",
        craft_duration_sec=1.6,
        tab_norm_coords=(0.38, 0.23),
        recipe_norm_coords=(0.37, 0.47),
        craft_all_norm_coords=(0.62, 0.74),
    ),
    "Calendula": CraftingRecipe(
        raw_material="Calendula",
        output_item="Calendula Panacea",
        threshold=20,
        batch_input_count=10,
        output_quantity=5,
        category="alchemy",
        window_hotkey="p",
        craft_duration_sec=1.6,
        tab_norm_coords=(0.38, 0.23),
        recipe_norm_coords=(0.37, 0.51),
        craft_all_norm_coords=(0.62, 0.74),
    ),
    "Mela": CraftingRecipe(
        raw_material="Mela",
        output_item="Mela Fruit Tart",
        threshold=20,
        batch_input_count=10,
        output_quantity=5,
        category="cooking",
        window_hotkey="p",
        craft_duration_sec=1.5,
        tab_norm_coords=(0.44, 0.23),
        recipe_norm_coords=(0.37, 0.36),
        craft_all_norm_coords=(0.62, 0.74),
    ),
    "Sapphire Gemstone": CraftingRecipe(
        raw_material="Sapphire Gemstone",
        output_item="Polished Sapphire",
        threshold=20,
        batch_input_count=10,
        output_quantity=5,
        category="jewelcrafting",
        window_hotkey="p",
        craft_duration_sec=1.8,
        tab_norm_coords=(0.50, 0.23),
        recipe_norm_coords=(0.37, 0.36),
        craft_all_norm_coords=(0.62, 0.74),
    ),
    "Ruby Gemstone": CraftingRecipe(
        raw_material="Ruby Gemstone",
        output_item="Polished Ruby",
        threshold=20,
        batch_input_count=10,
        output_quantity=5,
        category="jewelcrafting",
        window_hotkey="p",
        craft_duration_sec=1.8,
        tab_norm_coords=(0.50, 0.23),
        recipe_norm_coords=(0.37, 0.40),
        craft_all_norm_coords=(0.62, 0.74),
    ),
    "Diamond Gemstone": CraftingRecipe(
        raw_material="Diamond Gemstone",
        output_item="Brilliant Diamond",
        threshold=20,
        batch_input_count=10,
        output_quantity=5,
        category="jewelcrafting",
        window_hotkey="p",
        craft_duration_sec=2.0,
        tab_norm_coords=(0.50, 0.23),
        recipe_norm_coords=(0.37, 0.44),
        craft_all_norm_coords=(0.62, 0.74),
    ),
    "Yggdrasil Log": CraftingRecipe(
        raw_material="Yggdrasil Log",
        output_item="Yggdrasil Composite Plank",
        threshold=20,
        batch_input_count=10,
        output_quantity=5,
        category="woodworking",
        window_hotkey="p",
        craft_duration_sec=1.7,
        tab_norm_coords=(0.56, 0.23),
        recipe_norm_coords=(0.37, 0.36),
        craft_all_norm_coords=(0.62, 0.74),
    ),
    "Asvata Log": CraftingRecipe(
        raw_material="Asvata Log",
        output_item="Asvata Reinforced Timber",
        threshold=20,
        batch_input_count=10,
        output_quantity=5,
        category="woodworking",
        window_hotkey="p",
        craft_duration_sec=1.7,
        tab_norm_coords=(0.56, 0.23),
        recipe_norm_coords=(0.37, 0.40),
        craft_all_norm_coords=(0.62, 0.74),
    ),
    "Odyle": CraftingRecipe(
        raw_material="Odyle",
        output_item="Refined Odyle Crystal",
        threshold=20,
        batch_input_count=10,
        output_quantity=5,
        category="morphing",
        window_hotkey="k",
        craft_duration_sec=1.9,
        tab_norm_coords=(0.35, 0.25),
        recipe_norm_coords=(0.38, 0.37),
        craft_all_norm_coords=(0.62, 0.74),
    ),
    "Aether": CraftingRecipe(
        raw_material="Aether",
        output_item="Condensed Aether Crystal",
        threshold=20,
        batch_input_count=10,
        output_quantity=5,
        category="morphing",
        window_hotkey="k",
        craft_duration_sec=1.8,
        tab_norm_coords=(0.35, 0.25),
        recipe_norm_coords=(0.38, 0.41),
        craft_all_norm_coords=(0.62, 0.74),
    ),
}


def _known_unit_price(price_scanner: Optional[Any], material_name: str) -> Optional[int]:
    """
    Bilinen materyalin birim Kinah fiyatini dondurur; fiyat tablosunda
    olmayan (bos olmayan ama bilinmeyen) ad icin None dondurur (S-0602).
    Bilinmeyen ad hicbir zaman uydurma varsayilan fiyatla
    degerlenmez; cagiran, None durumunda kalemi rota/kazanc hesabindan
    cikarir ve `unpriced_materials` altinda raporlar.
    """
    if not material_name:
        return None
    clean = str(material_name).strip()
    if not clean:
        return None
    if price_scanner is not None:
        has_price = getattr(price_scanner, "has_price", None)
        if callable(has_price):
            try:
                if has_price(clean):
                    return int(price_scanner.get_price(clean, 0))
            except Exception:
                pass
            return None
        prices = getattr(price_scanner, "prices", None)
        if isinstance(prices, dict):
            if clean in prices:
                return int(prices[clean])
            clean_lower = clean.lower()
            for k, v in prices.items():
                if str(k).lower() == clean_lower:
                    return int(v)
            return None
        # get_price'li ama tablo erisimi olmayan tarayicilarda varsayilan
        # uydurma olacagi icin yalnizca varsayilan tablodaki adlar bilinir.
    if clean in DEFAULT_BROKER_PRICES:
        return int(DEFAULT_BROKER_PRICES[clean])
    clean_lower = clean.lower()
    for k, v in DEFAULT_BROKER_PRICES.items():
        if k.lower() == clean_lower:
            return int(v)
    return None


def calculate_net_hourly_kinah(
    gathered_counts: Dict[str, int],
    consumables_used: Dict[str, int],
    elapsed_hours: float = 1.0,
    price_scanner: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Toplanan materyallerin brüt Kinah değerinden tüketilebilir (iksir/parşömen)
    maliyetini düşerek saatlik net Kinah kârını hesaplar.

    Tüketilebilir maliyeti de brüt değerle AYNI tek kaynaktan (broker fiyat
    tablosu / canlı tarayıcı) çözülür (P0-7 / K-12): ikinci bir maliyet
    tablosu yoktur. Fiyatı bilinmeyen tüketilebilir uydurma fiyatla
    maliyetlendirilmez — 0 sayılır ve `unpriced_consumables` altında raporlanır.

    `price_scanner` bir TradeBrokerPriceScanner (get_price'li) ya da bir
    SessionProfitSimulator (``.price_scanner``'li) olabilir; hiçbiri verilmezse
    varsayılan fiyat tablosu kullanılır.
    """
    scanner = None
    tax_rate = 0.0
    if price_scanner is not None:
        if hasattr(price_scanner, "get_price"):
            scanner = price_scanner
        elif hasattr(price_scanner, "price_scanner"):
            scanner = getattr(price_scanner, "price_scanner")
        tax_rate = float(getattr(price_scanner, "broker_tax_rate", 0.0) or 0.0)

    unpriced_materials: List[str] = []

    def _unit_price(name: str) -> int:
        # Bilinmeyen ad uydurma 12000 Kinah ile degerlenmez (S-0602):
        # fiyatsiz kalem brüte 0 katki yapar ve unpriced_materials'de raporlanir.
        known = _known_unit_price(scanner, name)
        if known is None:
            if str(name) not in unpriced_materials:
                unpriced_materials.append(str(name))
            return 0
        return int(known)

    gross_kinah = 0
    for mat_name, qty in (gathered_counts or {}).items():
        if qty and int(qty) > 0:
            gross_kinah += int(qty) * _unit_price(str(mat_name))

    consumable_cost = 0
    unpriced_consumables: List[str] = []
    for item_name, qty in (consumables_used or {}).items():
        if qty and int(qty) > 0:
            # P0-7 / K-12: tuketilebilir maliyeti de TEK kaynaktan (broker
            # fiyat tablosu / canli tarayici) cozulur; ikinci fiyat tablosu
            # silindi. Bilinmeyen tuketilebilir uydurma fiyatla
            # maliyetlendirilmez: 0 sayilir, unpriced_consumables'de raporlanir.
            known_cost = _known_unit_price(scanner, str(item_name))
            if known_cost is None:
                if str(item_name) not in unpriced_consumables:
                    unpriced_consumables.append(str(item_name))
                continue
            consumable_cost += int(qty) * int(known_cost)

    hours = float(elapsed_hours) if elapsed_hours else 0.0
    if hours <= 0.0:
        hours = 1.0 / 3600.0
    tax_multiplier = max(0.0, min(1.0, 1.0 - tax_rate))
    net_kinah = int(round((gross_kinah - consumable_cost) * tax_multiplier))
    if unpriced_materials:
        logger.warning(
            "calculate_net_hourly_kinah: fiyati bilinmeyen materyal(ler) "
            "brüt kazanca katilmadi (uydurma fiyat yok): %s",
            ", ".join(sorted(unpriced_materials)),
        )
    if unpriced_consumables:
        logger.warning(
            "calculate_net_hourly_kinah: fiyati bilinmeyen tuketilebilir(ler) "
            "maliyete katilmadi (uydurma fiyat yok): %s",
            ", ".join(sorted(unpriced_consumables)),
        )
    return {
        "gross_kinah": int(gross_kinah),
        "consumable_cost_kinah": int(consumable_cost),
        "net_kinah": int(net_kinah),
        "elapsed_hours": float(hours),
        "net_kinah_per_hour": int(round(net_kinah / hours)),
        "formatted_net_per_hour": format_kinah_tr(int(round(net_kinah / hours))),
        "unpriced_materials": sorted(unpriced_materials),
        "unpriced_consumables": sorted(unpriced_consumables),
    }


# ==============================================================================
# 1. TRADE BROKER PRICE SCANNER & PROFITABLE ROUTE SELECTOR
# ==============================================================================
class TradeBrokerPriceScanner:
    """
    Oyun içi Pazar (Trade Broker) penceresini Bilgisayarlı Görü ile tarayan,
    `config/broker_prices.json` altında kalıcı fiyat veritabanı tutan ve
    `RouteManager` içerisindeki rotaların saatlik beklenen Kinah getirisini
    hesaplayarak en kârlı rotayı seçen sınıf.
    """

    # Pazar penceresi normalize ROI koordinatları (1920x1080 ve 1280x720 uyumlu)
    BROKER_WINDOW_ROI: Tuple[float, float, float, float] = (0.20, 0.16, 0.84, 0.84)
    BROKER_ROW_COUNT: int = 8

    def __init__(
        self,
        price_db_path: str = "config/broker_prices.json",
        auto_save: bool = True,
    ) -> None:
        # P3-6: göreli fiyat-veritabanı yolu proje köküne göre çözülür
        # (eski: os.path.abspath -> CWD'ye göre; bot başka dizinden
        # başlatılınca yanlış config/broker_prices.json okunur/yazılırdı).
        # Mutlak yol aynen geçer; hem okuma (load_prices) hem yazma
        # (_save_prices_unlocked) bu tek yoldan gider.
        self.price_db_path: str = os.fspath(resolve_project_path(price_db_path))
        self.auto_save: bool = bool(auto_save)
        self._lock = threading.Lock()
        self.prices: Dict[str, int] = copy.deepcopy(DEFAULT_BROKER_PRICES)
        # Kucuk-harfe cevrilmis O(1) fiyat dizini: get_price buyuk/kucuk harf
        # duyarsiz aramada tum sozlugu taramak yerine bu dizine bakar.
        # Ayni kucuk harfe sahip birden fazla anahtar varsa sozluk sirasinda
        # ilk gelen kazanir (eski dogrusal taramayla ayni sonuc).
        self._prices_lower: Dict[str, int] = {}
        self._rebuild_lower_index()
        self.last_scan_timestamp: float = 0.0
        self.scan_count: int = 0
        self.last_scan_detected: List[str] = []
        self.price_history: Dict[str, List[Dict[str, Union[int, float]]]] = {}

        self.load_prices()

    def _rebuild_lower_index(self) -> None:
        """``self.prices`` uzerinden kucuk-harf fiyat dizinini bastan kurar.

        Kilit altinda cagrilmalidir (``__init__``/``load_prices`` icinde).
        Yalnizca yazma yolunda (yukleme/guncelleme) calisir; okuma yolu
        (``get_price``) dizini degistirmez.
        """
        index: Dict[str, int] = {}
        for key, value in self.prices.items():
            index.setdefault(key.lower(), int(value))
        self._prices_lower = index

    def _refresh_lower_index_entry(self, material_name: str) -> None:
        """Tek bir materyalin dizin girdisini guncel fiyatlarla tazeler.

        ``update_price`` sonrasi cagrilir; ayni kucuk harfe sahip anahtarlar
        arasinda sozluk sirasinda ilk gelenin fiyati dizine yazilir.
        Kilit altinda cagrilmalidir.
        """
        clean_lower = str(material_name).strip().lower()
        for key, value in self.prices.items():
            if key.lower() == clean_lower:
                self._prices_lower[clean_lower] = int(value)
                return
        self._prices_lower.pop(clean_lower, None)

    # ------------------------------------------------------------------
    # Kalıcı Fiyat Veritabanı Yönetimi (config/broker_prices.json)
    # ------------------------------------------------------------------
    def load_prices(self) -> Dict[str, int]:
        """
        `config/broker_prices.json` dosyasından fiyatları yükler.
        Dosya yoksa varsayılan Aion 2 materyal fiyatlarıyla oluşturur.
        """
        with self._lock:
            if os.path.isfile(self.price_db_path):
                try:
                    with open(self.price_db_path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    loaded_map = raw.get("prices", raw) if isinstance(raw, dict) else {}
                    if isinstance(loaded_map, dict):
                        for mat_name, val in loaded_map.items():
                            if isinstance(val, (int, float)) and val > 0:
                                self.prices[str(mat_name)] = int(round(val))
                    # Zaman damgasi bozuksa yalnizca o alan yoksayilir; fiyatlar
                    # yuklenmisken 'varsayilanlara donuldu' denmez (F-0433).
                    if isinstance(raw, dict) and "last_scan_timestamp" in raw:
                        try:
                            self.last_scan_timestamp = float(raw["last_scan_timestamp"])
                        except (TypeError, ValueError):
                            logger.warning(
                                "Pazar fiyat veritabaninda gecersiz last_scan_timestamp "
                                "yoksayildi (%s); fiyatlar yuklendigi haliyle kullaniliyor.",
                                self.price_db_path,
                            )
                except Exception as exc:
                    logger.warning(
                        "Pazar fiyat veritabanı okunamadı (%s), varsayılanlar kullanılıyor: %s",
                        self.price_db_path,
                        exc,
                    )
            else:
                if self.auto_save:
                    self._save_prices_unlocked()
            self._rebuild_lower_index()
            return dict(self.prices)

    def _save_prices_unlocked(self) -> bool:
        """Fiyat veritabanini atomik yazar: gecici dosyaya yaz + os.replace (F-0257)."""
        try:
            parent_dir = os.path.dirname(self.price_db_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            payload = {
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "last_scan_timestamp": float(self.last_scan_timestamp),
                "scan_count": int(self.scan_count),
                "currency": "Kinah",
                "prices": {k: int(v) for k, v in sorted(self.prices.items())},
            }
            tmp_path = self.price_db_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.price_db_path)
            return True
        except Exception as exc:
            logger.error("Pazar fiyat veritabanı kaydedilemedi (%s): %s", self.price_db_path, exc)
            return False

    def save_prices(self) -> bool:
        """Güncel materyal fiyatlarını `config/broker_prices.json` dosyasına yazar."""
        with self._lock:
            return self._save_prices_unlocked()

    def get_price(
        self, material_name: str, default: Optional[int] = None
    ) -> Optional[int]:
        """
        Belirtilen materyalin güncel Kinah birim fiyatını döndürür.
        Yalnizca tam eslesme (buyuk/kucuk harf duyarsiz): kismi eslesme
        'Pure Odyle' gibi ust seviye materyali 'Odyle' fiyatiyla
        fiyatliyordu (F-0087), o yuzden kaldirildi.
        P0-7 / K-12: bilinmeyen ad icin artik uydurma 10000 Kinah
        dondurulmez — `default` verilmediyse None doner (bilinmiyor).
        Acik `default` veren cagiranlar (or. `get_price(..., 0)`) eski
        davranisi korur: bilinmeyen ad o degeri alir. None donen cagiran,
        sonucu sayi gibi kullanmadan once kontrol etmelidir.
        """
        if not material_name:
            return default
        clean = str(material_name).strip()
        if not clean:
            return default
        with self._lock:
            if clean in self.prices:
                return int(self.prices[clean])
            # Buyuk/kucuk harf duyarsiz arama: dogrusal tarama yerine O(1)
            # dizin (eski davranisla ayni sonuc: ilk eslesen anahtar kazanir).
            hit = self._prices_lower.get(clean.lower())
            if hit is not None:
                return int(hit)
        return default

    def has_price(self, material_name: str) -> bool:
        """
        Materyal adinin fiyat tablosunda bilindigini soyler (S-0602).
        `get_price` bilinmeyen ada `default` verilmediyse None dondurur
        (P0-7); bilinirlik gerekiyorsa bu metod kullanilmalidir.
        Bilinmeyen ad hicbir hesapta uydurma fiyatla degerlenmez.
        """
        if not material_name:
            return False
        clean = str(material_name).strip()
        if not clean:
            return False
        with self._lock:
            if clean in self.prices:
                return True
            clean_lower = clean.lower()
            return any(k.lower() == clean_lower for k in self.prices)

    def update_price(self, material_name: str, unit_price: int, save: bool = True) -> int:
        """Tek bir materyalin birim Kinah fiyatını günceller ve geçmişe kaydeder."""
        clean_name = str(material_name).strip()
        if not clean_name:
            raise ValueError("update_price: materyal adi bos olamaz")
        try:
            numeric_price = float(unit_price)
        except (TypeError, ValueError):
            raise ValueError(
                f"update_price: sayisal olmayan fiyat '{unit_price!r}'"
            ) from None
        if not math.isfinite(numeric_price):
            raise ValueError("update_price: fiyat sonlu bir sayi olmali")
        clamped_price = max(1, int(round(numeric_price)))
        now = time.time()
        with self._lock:
            self.prices[clean_name] = clamped_price
            self._refresh_lower_index_entry(clean_name)
            hist = self.price_history.setdefault(clean_name, [])
            hist.append({"timestamp": now, "price": clamped_price})
            if len(hist) > 100:
                del hist[:-100]
            # Acik 'save=True' istegi auto_save=False iken bile yazilir;
            # sessizce yoksayilmaz (F-0258). Otomatik kayitlar ayri tutulur.
            if save:
                self._save_prices_unlocked()
        return clamped_price

    # ------------------------------------------------------------------
    # Bilgisayarlı Görü (CV) ile Trade Broker Penceresi Tarama
    # ------------------------------------------------------------------
    def _estimate_row_price_from_digits_roi(
        self,
        price_roi_bgr: np.ndarray,
        _baseline_price: int,
    ) -> Optional[int]:
        """
        Pazar satirinin fiyat sutunundan birim fiyat OKUNAMAZ: repoda OCR
        motoru yok, piksel yogunlugundan turetilen 'modulation' uydurma
        fiyatti (F-0014). Bu yuzden hicbir zaman tahmini fiyat uretilmez;
        None doner ve tarama fiyati degistirmez. OCR eklendiginde bu
        fonksiyon gercek rakam okuma ile doldurulmalidir.
        """
        if price_roi_bgr is None or price_roi_bgr.size == 0:
            return None
        # OCR olmadan rakam pikseli saymak fiyat olcmez; hicbir deger
        # uretilmez. OCR eklendiginde bu fonksiyon gercek rakam okuma ile
        # doldurulmalidir.
        return None

    def _classify_row_material(self, icon_roi_bgr: np.ndarray, _row_index: int) -> Optional[str]:
        """Pazar satırının solundaki materyal ikon kesitini HSV imzasına göre sınıflandırır."""
        if icon_roi_bgr is None or icon_roi_bgr.size == 0:
            return None

        hsv = cv2.cvtColor(icon_roi_bgr, cv2.COLOR_BGR2HSV)
        best_material: Optional[str] = None
        best_ratio: float = 0.06  # En az %6 aktif ikon pikseli eşleşmeli

        total_px = float(max(1, hsv.shape[0] * hsv.shape[1]))
        for mat_name, lower_arr, upper_arr in BROKER_ICON_HSV_SIGNATURE_ARRAYS:
            mask = cv2.inRange(hsv, lower_arr, upper_arr)
            ratio = float(cv2.countNonZero(mask)) / total_px
            if ratio > best_ratio:
                best_ratio = ratio
                best_material = mat_name

        if best_material is not None:
            return best_material
        # Satir indeksinden materyal uydurma kaldirildi (F-0014): HSV imzasi
        # tutmayan satir bilinmeyen sayilir, siradaki materyal atanmaz.
        return None

    def scan_broker_window(self, frame: np.ndarray) -> Dict[str, int]:
        """
        Oyun ici Trade Broker (Pazar) penceresini tarar.

        DURUSTLUK SINIRI (F-0014): Repoda OCR motoru olmadigi icin bu tarama
        fiyat OLCEMEZ; fiyatlari uydurmaz ve degistirmez, yalnizca HSV
        imzasina gore materyal tespit edildigini raporlar
        (``last_scan_detected``) ve onbellek fiyatlarini oldugu gibi
        dondurur. Gercek fiyat okuma OCR entegrasyonu gerektirir.

        TARAMA KAYDI (S-0702): Gecerli bir karede tarama gercekten
        calistiysa ``scan_count`` 1 artar ve ``price_history``'ye kayit
        duser: tespit edilen her materyal icin o anki ONBELLEK fiyati
        ``source="cache"``, ``measured=False`` isaretli kayit olarak
        yazilir (olcum degildir, fiyat degistirmez); ayrica her tamamlanan
        tarama ``price_history["__scan__"]`` altina bir tarama olayi
        olarak kaydedilir (tarama olayidir, fiyat gozlemi degildir).
        Gecersiz/bos karede veya pazar penceresi tespit edilemediginde
        tarama yapilmis sayilmaz: ``scan_count`` artmaz, kayit yazilmaz.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("scan_broker_window: Gecersiz veya bos goruntu karesi, onbellek fiyatlari donduruluyor.")
            return dict(self.prices)

        # 1 boyutlu / beklenmeyen sekilli kareler IndexError ile cokmesin (F-0089).
        if frame.ndim == 2:
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            bgr = frame
        else:
            logger.warning(
                "scan_broker_window: Desteklenmeyen kare sekli %s, onbellek fiyatlari donduruluyor.",
                getattr(frame, "shape", None),
            )
            return dict(self.prices)

        h, w = bgr.shape[:2]
        x1 = max(0, min(w - 1, int(round(w * self.BROKER_WINDOW_ROI[0]))))
        y1 = max(0, min(h - 1, int(round(h * self.BROKER_WINDOW_ROI[1]))))
        x2 = max(x1 + 1, min(w, int(round(w * self.BROKER_WINDOW_ROI[2]))))
        y2 = max(y1 + 1, min(h, int(round(h * self.BROKER_WINDOW_ROI[3]))))

        broker_roi = bgr[y1:y2, x1:x2]
        roi_h, roi_w = broker_roi.shape[:2]

        # Pazar penceresi kontrolu (F-0014): tek renk / bos kare pazar
        # penceresi degildir; varyans yoksa tarama yapilmis sayilmaz.
        # _pooled_std_bgr, np.std ile ayni degeri OpenCV meanStdDev ile
        # ~2-4.5x daha hizli hesaplar (tarama basina ~5-40 ms kazanc).
        if roi_h < 24 or roi_w < 24 or _pooled_std_bgr(broker_roi) < 5.0:
            logger.warning(
                "scan_broker_window: Pazar penceresi tespit edilemedi "
                "(ROI varyansi cok dusuk), fiyatlar degistirilmedi."
            )
            return dict(self.prices)

        # Tablo satirlari bolgesi (ust baslik cubugunu ve alt sayfalama barini haric tut)
        table_top = int(round(roi_h * 0.14))
        table_bottom = int(round(roi_h * 0.88))
        table_h = max(1, table_bottom - table_top)
        row_h = max(1, table_h // self.BROKER_ROW_COUNT)

        detected_materials: List[str] = []

        for row_idx in range(self.BROKER_ROW_COUNT):
            ry1 = table_top + row_idx * row_h
            ry2 = min(table_bottom, ry1 + row_h)
            if ry2 - ry1 < 6:
                continue

            row_img = broker_roi[ry1:ry2, :]
            # Sol ikon sutunu (%4 .. %16 genislik)
            ix1 = int(round(roi_w * 0.04))
            ix2 = max(ix1 + 4, int(round(roi_w * 0.16)))
            icon_crop = row_img[:, ix1:ix2]

            detected_mat = self._classify_row_material(icon_crop, row_idx)
            if detected_mat:
                detected_materials.append(detected_mat)

        self.last_scan_detected = list(detected_materials)
        # S-0702: Tarama gercekten calisti -> sayac artar ve price_history
        # kaydi duser. F-0014 korunur: fiyat olculmez/degistirilmez;
        # tespit edilen materyal icin yalnizca o anki onbellek fiyati
        # "cache" kaynakli, olculmemis kayit olarak yazilir. Hic materyal
        # tespit edilemese bile tamamlanan tarama "__scan__" altinda
        # olay olarak kaydedilir (fiyat gozlemi degildir).
        now = time.time()
        with self._lock:
            self.scan_count += 1
            for mat_name in detected_materials:
                cached_price = self.prices.get(mat_name)
                if cached_price is None:
                    continue
                hist = self.price_history.setdefault(mat_name, [])
                hist.append(
                    {
                        "timestamp": now,
                        "price": int(cached_price),
                        "source": "cache",
                        "measured": False,
                    }
                )
                if len(hist) > 100:
                    del hist[:-100]
            scan_log = self.price_history.setdefault("__scan__", [])
            scan_log.append(
                {
                    "timestamp": now,
                    "scan_count": int(self.scan_count),
                    "detected": list(detected_materials),
                }
            )
            if len(scan_log) > 100:
                del scan_log[:-100]
        if detected_materials:
            logger.info(
                "Pazar (Trade Broker) taramasi: %d satirda materyal tespit edildi "
                "(%s); OCR olmadigi icin fiyatlar guncellenmedi, onbellek fiyatlari donduruluyor.",
                len(detected_materials),
                ", ".join(detected_materials),
            )
        else:
            logger.info(
                "Pazar (Trade Broker) taramasi: hicbir materyal tespit edilemedi; "
                "fiyatlar guncellenmedi."
            )
        return dict(self.prices)


    def evaluate_route_profitability(
        self,
        nodes: List[Any],
        region_name: str,
        route_name: str,
        meters_per_pixel: float = WORLD_MAP_METERS_PER_PX,
        mount_speed_mps: float = 8.5,
        harvest_stage_sec: float = 2.4,
    ) -> Dict[str, Any]:
        """
        Verilen düğüm listesi için:
        - Materyal dağılımını (`material_composition`)
        - Düğüm başı 3x hasat çarpanını (`max_harvests`, varsayılan 3)
        - Döngü başına toplam Kinah getirisini (`expected_kinah_per_cycle`)
        - Düğümler arası mesafe + hasat süresi + hava/uçuş ek süresini hesaba katarak
          saatlik beklenen Kinah/Altın getirisini (`expected_kinah_per_hour`) hesaplar.
        """
        if not nodes:
            return {
                "region": region_name,
                "route": route_name,
                "total_nodes": 0,
                "total_harvests_per_cycle": 0,
                "expected_kinah_per_cycle": 0,
                "estimated_cycle_seconds": 0.0,
                "expected_kinah_per_hour": 0,
                "material_composition": {},
                "material_kinah_breakdown": {},
                "unpriced_nodes_skipped": 0,
                "unknown_materials": [],
            }

        material_composition: Dict[str, int] = {}
        material_kinah_breakdown: Dict[str, int] = {}
        total_harvests_per_cycle = 0
        priced_nodes = 0
        unpriced_nodes_skipped = 0
        unknown_materials: List[str] = []
        expected_kinah_per_cycle = 0
        total_gather_seconds = 0.0
        total_travel_pixels = 0.0

        prev_x: Optional[float] = None
        prev_y: Optional[float] = None

        for node in nodes:
            if isinstance(node, dict):
                raw_res = node.get("resource_type")
                max_h = max(1, int(node.get("max_harvests", 3) or 3))
                in_air = bool(node.get("in_air", False))
                mx = float(node.get("map_x", 0) or 0)
                my = float(node.get("map_y", 0) or 0)
            else:
                raw_res = getattr(node, "resource_type", None)
                max_h = max(1, int(getattr(node, "max_harvests", 3) or 3))
                in_air = bool(getattr(node, "in_air", False))
                mx = float(getattr(node, "map_x", 0) or 0)
                my = float(getattr(node, "map_y", 0) or 0)

            # Turu bilinmeyen dugum sessizce en pahali cevher sayilmaz
            # (F-0259): fiyatlanamaz, rota karindan cikarilir.
            if not raw_res or not str(raw_res).strip():
                logger.warning(
                    "evaluate_route_profitability: resource_type olmayan dugum "
                    "atlandi (bolge=%s, rota=%s).",
                    region_name,
                    route_name,
                )
                continue
            res_type = str(raw_res).strip()

            # Fiyati bilinmeyen (bos olmayan) materyal de uydurma 12000
            # Kinah ile degerlenmez (S-0602, F-0259'un deligi): dugum
            # rota karindan cikarilir, adi raporlanir.
            unit_price = _known_unit_price(self, res_type)
            if unit_price is None:
                unpriced_nodes_skipped += 1
                if res_type not in unknown_materials:
                    unknown_materials.append(res_type)
                logger.warning(
                    "evaluate_route_profitability: fiyati bilinmeyen materyal "
                    "'%s' rota karindan cikarildi (bolge=%s, rota=%s).",
                    res_type,
                    region_name,
                    route_name,
                )
                continue
            priced_nodes += 1

            node_kinah = unit_price * max_h

            material_composition[res_type] = material_composition.get(res_type, 0) + max_h
            material_kinah_breakdown[res_type] = material_kinah_breakdown.get(res_type, 0) + node_kinah
            total_harvests_per_cycle += max_h
            expected_kinah_per_cycle += node_kinah

            # 3 aşamalı hasat süresi + hava düğümü ise uçuş/tırmanma ek süresi
            node_harvest_time = (max_h * harvest_stage_sec) + (2.8 if in_air else 0.8)
            total_gather_seconds += node_harvest_time

            if prev_x is not None and prev_y is not None and (mx > 0 or my > 0):
                # Gercek mesafe kullanilir: 320 px kesme ve ilk dugume
                # +38 px ekleme kaldirildi (F-0090); ilk dugumde yol 0'dir.
                dist_px = math.hypot(mx - prev_x, my - prev_y)
                total_travel_pixels += dist_px
            prev_x, prev_y = mx, my

        travel_meters = total_travel_pixels * max(0.05, float(meters_per_pixel))
        travel_seconds = travel_meters / max(1.0, float(mount_speed_mps))
        # Bir döngünün tamamlanma süresi (en az 18 saniye)
        estimated_cycle_seconds = max(18.0, total_gather_seconds + travel_seconds)
        cycles_per_hour = 3600.0 / estimated_cycle_seconds
        expected_kinah_per_hour = int(round(expected_kinah_per_cycle * cycles_per_hour))

        return {
            "region": region_name,
            "route": route_name,
            "total_nodes": int(priced_nodes),
            "total_harvests_per_cycle": int(total_harvests_per_cycle),
            "expected_kinah_per_cycle": int(expected_kinah_per_cycle),
            "estimated_cycle_seconds": round(estimated_cycle_seconds, 2),
            "expected_kinah_per_hour": int(expected_kinah_per_hour),
            "formatted_kinah_per_hour": format_kinah_tr(expected_kinah_per_hour),
            "material_composition": material_composition,
            "material_kinah_breakdown": material_kinah_breakdown,
            "unpriced_nodes_skipped": int(unpriced_nodes_skipped),
            "unknown_materials": sorted(unknown_materials),
        }

    def select_most_profitable_route(
        self,
        route_manager: Any,
        region: Optional[str] = None,
        activate: bool = True,
    ) -> Dict[str, Any]:
        """
        `route_manager` içerisindeki mevcut tüm rotaları (veya belirtilen `region`
        altındaki rotaları) analiz eder, düğüm materyal kompozisyonu + 3x hasat
        çarpanı (`max_harvests`) + güncel Pazar birim fiyatlarına göre saatlik
        beklenen Kinah getirisini (`expected_kinah_per_hour`) hesaplar ve en kârlı
        rotayı otomatik olarak seçip etkinleştirir.
        """
        rankings: List[Dict[str, Any]] = []
        meters_per_px = float(getattr(route_manager, "meters_per_pixel", WORLD_MAP_METERS_PER_PX) or WORLD_MAP_METERS_PER_PX)

        # get_route_nodes imzasi rota basina degismez: her rota icin yeniden
        # inspect.signature cagirmak (40 rotada 40 kez) yerine bir kez hesaplanir.
        # Imza cikarilamazsa eski davranistaki gibi ayni hata yukari firlar
        # (hicbir rota degerlendirilmeden once).
        route_nodes_params = None
        if hasattr(route_manager, "get_route_nodes") and callable(route_manager.get_route_nodes):
            route_nodes_params = inspect.signature(route_manager.get_route_nodes).parameters

        # Taranacak bölgeleri belirle
        candidate_regions: List[str] = []
        if region:
            candidate_regions = [str(region)]
        elif hasattr(route_manager, "get_regions") and callable(route_manager.get_regions):
            candidate_regions = list(route_manager.get_regions())
        elif hasattr(route_manager, "_regions") and isinstance(route_manager._regions, dict):
            candidate_regions = list(route_manager._regions.keys())

        if not candidate_regions and hasattr(route_manager, "active_region"):
            candidate_regions = [str(route_manager.active_region or "Verteron")]

        for reg_name in candidate_regions:
            routes_in_reg: List[str] = ["All"]
            if hasattr(route_manager, "get_routes") and callable(route_manager.get_routes):
                try:
                    fetched_routes = route_manager.get_routes(reg_name)
                    if fetched_routes:
                        routes_in_reg = list(fetched_routes)
                except Exception:
                    routes_in_reg = ["All"]

            for rt_name in routes_in_reg:
                nodes: List[Any] = []
                if hasattr(route_manager, "get_route_nodes") and callable(route_manager.get_route_nodes):
                    # Imza once incelenir; ic TypeError imza uyumsuzlugu
                    # sanilip cagri tekrarlanmaz (F-0260). Imza dongu
                    # disinda bir kez hesaplandi; burada yalnizca kullanilir.
                    try:
                        if (
                            route_nodes_params is not None
                            and "region_name" in route_nodes_params
                            and "route_name" in route_nodes_params
                        ):
                            nodes = list(route_manager.get_route_nodes(region_name=reg_name, route_name=rt_name))
                        elif (
                            route_nodes_params is not None
                            and "region" in route_nodes_params
                            and "route" in route_nodes_params
                        ):
                            nodes = list(route_manager.get_route_nodes(region=reg_name, route=rt_name))
                        else:
                            nodes = list(route_manager.get_route_nodes(reg_name, rt_name))
                    except TypeError:
                        raise
                elif hasattr(route_manager, "get_nodes") and callable(route_manager.get_nodes):
                    nodes = list(route_manager.get_nodes(region=reg_name, route=rt_name))

                if not nodes:
                    continue

                eval_res = self.evaluate_route_profitability(
                    nodes=nodes,
                    region_name=reg_name,
                    route_name=rt_name,
                    meters_per_pixel=meters_per_px,
                )
                rankings.append(eval_res)

        # Rota yoksa sahte 'en karli rota' uretilmez (F-0015): hicbir sey
        # aktiflestirilmez, cagiran bos sonucu acikca gorur.
        if not rankings:
            logger.warning(
                "select_most_profitable_route: degerlendirilecek rota yok "
                "(bolge=%s); rota aktiflestirilmedi.",
                region,
            )
            return {
                "best_region": None,
                "best_route": None,
                "expected_kinah_per_hour": 0,
                "formatted_kinah_per_hour": "0",
                "expected_kinah_per_cycle": 0,
                "total_nodes": 0,
                "total_harvests_per_cycle": 0,
                "material_composition": {},
                "material_kinah_breakdown": {},
                "route_rankings": [],
                "activated": False,
                "error": "no_routes_available",
                "recommendation_tr": "Degerlendirilecek rota bulunamadi; hicbir rota aktiflestirilmedi.",
            }

        # Saatlik Kinah getirisine göre azalan sıralama (eşitlikte özel rota ismi 'All' önüne geçer)
        rankings.sort(
            key=lambda item: (
                int(item.get("expected_kinah_per_hour", 0)),
                1 if item.get("route") != "All" else 0,
                int(item.get("total_nodes", 0)),
            ),
            reverse=True,
        )

        best = rankings[0]
        best_region = str(best["region"])
        best_route = str(best["route"])
        best_kph = int(best["expected_kinah_per_hour"])

        activated = False
        if activate and route_manager is not None:
            try:
                if hasattr(route_manager, "active_region"):
                    route_manager.active_region = best_region
                if hasattr(route_manager, "active_route"):
                    route_manager.active_route = best_route
                if hasattr(route_manager, "_sync_legacy_data") and callable(route_manager._sync_legacy_data):
                    route_manager._sync_legacy_data()
                activated = True
            except Exception as exc:
                logger.warning("En kârlı rota aktifleştirilirken uyarı: %s", exc)

        formatted_kph = format_kinah_tr(best_kph)
        recommendation_tr = (
            f"En Kârlı Rota: {best_region} / {best_route} "
            f"(Beklenen: {formatted_kph} Kinah/saat, {best['total_nodes']} düğüm, "
            f"{best['total_harvests_per_cycle']}x hasat/döngü)"
        )
        logger.info("[EconomyBroker] %s", recommendation_tr)

        return {
            "best_region": best_region,
            "best_route": best_route,
            "expected_kinah_per_hour": best_kph,
            "formatted_kinah_per_hour": formatted_kph,
            "expected_kinah_per_cycle": int(best["expected_kinah_per_cycle"]),
            "total_nodes": int(best["total_nodes"]),
            "total_harvests_per_cycle": int(best["total_harvests_per_cycle"]),
            "material_composition": dict(best["material_composition"]),
            "material_kinah_breakdown": dict(best["material_kinah_breakdown"]),
            "route_rankings": rankings,
            "activated": activated,
            "recommendation_tr": recommendation_tr,
        }

    def get_economy_broker_ready_flag(self) -> dict:
        return {"module": "economy_broker", "ready": bool(validate_economy_broker_contract())}

    def get_economy_broker_version_info(self) -> dict:
        return {"module": "economy_broker", "version": __version__, "ready": bool(validate_economy_broker_contract())}

    def calculate_net_hourly_kinah(
        self,
        gathered_counts: Dict[str, int],
        consumables_used: Dict[str, int],
        elapsed_hours: float = 1.0,
    ) -> Dict[str, Any]:
        """Seçili materyallere göre iksir/parşömen maliyetini düşerek saatlik net Kinah kârını hesaplar."""
        return calculate_net_hourly_kinah(
            gathered_counts=gathered_counts,
            consumables_used=consumables_used,
            elapsed_hours=elapsed_hours,
            price_scanner=self,
        )


# ==============================================================================
# 2. AUTO-CRAFTING & SUBSTANCE MORPHING QUEUE
# ==============================================================================
class AutoCraftingMorphQueue:
    """
    Toplanan ham materyal envanterini takip eden (`record_gathered_material`) ve
    belirlenen eşik değerine (örn. 20 veya 50 adet) ulaşan materyalleri
    Üretim / Dönüştürme (Crafting / Substance Morphing) penceresini açarak
    külçe, iksir, yemek veya dönüştürülmüş kristallere çeviren kuyruk yöneticisi.
    """

    def __init__(
        self,
        default_threshold: int = 20,
        recipes: Optional[Dict[str, CraftingRecipe]] = None,
        price_scanner: Optional[TradeBrokerPriceScanner] = None,
    ) -> None:
        self.default_threshold: int = max(1, int(default_threshold))
        self.price_scanner: Optional[TradeBrokerPriceScanner] = price_scanner
        # RLock: _find_recipe kilit altinda da cagrilabilir (F-0253).
        self._lock = threading.RLock()

        self.recipes: Dict[str, CraftingRecipe] = copy.deepcopy(DEFAULT_CRAFTING_RECIPES)
        if recipes:
            for k, v in recipes.items():
                self.recipes[str(k)] = v

        # Varsayılan eşik değerini tüm reçetelere uygula
        for recipe in self.recipes.values():
            recipe.threshold = self.default_threshold

        self.raw_inventory: Dict[str, int] = {}
        self.crafted_inventory: Dict[str, int] = {}
        self.pending_queue: List[Dict[str, Any]] = []
        self.crafting_history: List[Dict[str, Any]] = []
        # Son UI uretim sirasinin gercek sonucu (S-0601): True yalnizca
        # tum girdi adimlari hatasiz calistiysa; eksik yetenek / istisna
        # durumunda False olur ve process_crafting_queue envanteri
        # degistirmeden isi yeniden kuyruga alir.
        self.last_ui_crafting_ok: bool = False
        self.last_ui_crafting_error: Optional[str] = None

    def set_threshold(self, material_name: str, threshold: int) -> None:
        """Belirli bir ham materyalin otomatik üretim/dönüştürme eşik sayısını günceller."""
        clean = str(material_name).strip()
        new_thresh = max(1, int(threshold))
        with self._lock:
            if clean in self.recipes:
                self.recipes[clean].threshold = new_thresh
            else:
                self.recipes[clean] = CraftingRecipe(
                    raw_material=clean,
                    output_item=f"Refined {clean}",
                    threshold=new_thresh,
                    batch_input_count=max(1, min(10, new_thresh)),
                    output_quantity=max(1, min(5, new_thresh // 2)),
                    category="morphing",
                )

    def _find_recipe(self, material_name: str) -> CraftingRecipe:
        """Recete bul/olustur; sozluk degisikligi kilit altinda yapilir (F-0253)."""
        clean = str(material_name).strip()
        with self._lock:
            if clean in self.recipes:
                return self.recipes[clean]
            clean_lower = clean.lower()
            for k, r in self.recipes.items():
                if k.lower() == clean_lower:
                    return r
            # Dinamik reçete oluştur (listede olmayan yeni bir Aion 2 materyali ise)
            dynamic_recipe = CraftingRecipe(
                raw_material=clean,
                output_item=f"Refined {clean}",
                threshold=self.default_threshold,
                batch_input_count=min(10, self.default_threshold),
                output_quantity=max(1, min(5, self.default_threshold // 2)),
                category="morphing",
                window_hotkey="k",
            )
            self.recipes[clean] = dynamic_recipe
            return dynamic_recipe

    def record_gathered_material(
        self,
        material_name: str,
        count: int = 1,
    ) -> Dict[str, Any]:
        """
        Toplanan ham materyali envantere ekler.
        Materyal miktarı reçete eşik değerine (`threshold`, örn. 20 veya 50) ulaştığında
        otomatik olarak üretim/dönüştürme kuyruğuna (`pending_queue`) görev ekler.
        """
        if not material_name or int(count) <= 0:
            return {"material": str(material_name), "current_count": 0, "queued": False}

        recipe = self._find_recipe(material_name)
        canonical_name = recipe.raw_material
        added = int(count)

        with self._lock:
            current_total = self.raw_inventory.get(canonical_name, 0) + added
            self.raw_inventory[canonical_name] = current_total

            already_queued = any(
                item.get("raw_material") == canonical_name and item.get("status") == "queued"
                for item in self.pending_queue
            )
            ready_to_craft = current_total >= recipe.threshold
            newly_queued = False
            if ready_to_craft and not already_queued:
                batches = max(1, current_total // max(1, recipe.batch_input_count))
                job = {
                    "job_id": f"craft_{int(time.time() * 1000)}_{len(self.crafting_history) + len(self.pending_queue) + 1}",
                    "raw_material": canonical_name,
                    "output_item": recipe.output_item,
                    "category": recipe.category,
                    "threshold": recipe.threshold,
                    "available_raw_count": current_total,
                    "batches": batches,
                    "window_hotkey": recipe.window_hotkey,
                    "status": "queued",
                    "queued_at": time.time(),
                }
                self.pending_queue.append(job)
                newly_queued = True
                logger.info(
                    "[AutoCrafting] '%s' eşik değerine ulaştı (%d/%d) -> '%s' üretimi kuyruğa eklendi (%d parti).",
                    canonical_name,
                    current_total,
                    recipe.threshold,
                    recipe.output_item,
                    batches,
                )

            return {
                "material": canonical_name,
                "current_count": current_total,
                "threshold": recipe.threshold,
                "output_item": recipe.output_item,
                # 'queued' bu cagrida yeni is eklendi mi demektir; is zaten
                # kuyruktayken ikinci kayit True donmez (F-0254).
                "queued": newly_queued,
                "pending_queue_size": len(self.pending_queue),
            }

    def _execute_ui_crafting_sequence(
        self,
        recipe: CraftingRecipe,
        input_ctrl: Optional[Any],
        test_mode: bool,
    ) -> List[str]:
        """
        Üretim / Substance Morphing penceresini açar, kategori sekmesini ve reçeteyi seçer,
        'Craft All / Morph' butonuna tıklar ve pencereyi kapatır.

        DONUS + `last_ui_crafting_ok` (S-0601): donen adim listesi tek
        basina basari kaniti degildir. Gercek modda tum adimlar hatasiz
        calistiysa `last_ui_crafting_ok=True`; girdi denetleyicisi tiklama
        yeteneginden yoksunsa veya bir adim istisna firlatirsa hicbir
        girdi gonderilmeden / kismi adimlarla `last_ui_crafting_ok=False`
        olur ve `last_ui_crafting_error` nedeni tasir — tiklamalar sessizce
        atlanmaz, cagiran 'completed' yazamaz.
        """
        executed_steps: List[str] = []
        hotkey = str(recipe.window_hotkey or "p").lower()
        self.last_ui_crafting_ok = False
        self.last_ui_crafting_error = None

        if input_ctrl is not None and not test_mode:
            # Tiklama yetenegi olmayan denetleyiciyle pencere acmak hayali
            # uretimdir: hicbir girdi gonderilmeden basarisiz don (S-0601).
            # Interception donanim yolu sarttir; eksik yetenek sessiz
            # soft dususle kapatilmaz.
            missing_caps = [
                cap
                for cap in ("press_key", "denormalize_coords", "human_click")
                if not hasattr(input_ctrl, cap)
            ]
            if missing_caps:
                self.last_ui_crafting_error = (
                    "input_ctrl eksik yetenek: " + ", ".join(missing_caps)
                )
                logger.error(
                    "AutoCrafting UI sirasi baslatilmadi: %s; hicbir tik "
                    "gonderilmedi, uretim yapilmadi.",
                    self.last_ui_crafting_error,
                )
                return executed_steps
            try:
                input_ctrl.press_key(hotkey)
                executed_steps.append(f"press_key('{hotkey}')")
                # Pencere acilisi beklenir (F-0084).
                time.sleep(0.5)

                tx, ty = input_ctrl.denormalize_coords(*recipe.tab_norm_coords)
                input_ctrl.human_click(tx, ty)
                executed_steps.append(f"click_tab({tx}, {ty})")
                time.sleep(0.3)

                rx, ry = input_ctrl.denormalize_coords(*recipe.recipe_norm_coords)
                input_ctrl.human_click(rx, ry)
                executed_steps.append(f"click_recipe({rx}, {ry})")
                time.sleep(0.3)

                cx, cy = input_ctrl.denormalize_coords(*recipe.craft_all_norm_coords)
                input_ctrl.human_click(cx, cy)
                executed_steps.append(f"click_craft_all({cx}, {cy})")
                # Uretim suresi beklenir; craft_duration_sec artik kullanilir.
                time.sleep(max(0.0, float(recipe.craft_duration_sec)))

                input_ctrl.press_key("esc")
                executed_steps.append("press_key('esc')")
                self.last_ui_crafting_ok = True
            except Exception as exc:
                # Kismi/bos adimlar basari sayilmaz (S-0601): istisna
                # yutulup warning ile gecilmez, hata olarak isaretlenir.
                self.last_ui_crafting_ok = False
                self.last_ui_crafting_error = str(exc)
                logger.error(
                    "AutoCrafting donanım girdi adımı BASARISIZ: %s; "
                    "uretim tamamlanmadi sayilir.",
                    exc,
                )
        else:
            # Test modu veya sanal yürütme
            if input_ctrl is not None and hasattr(input_ctrl, "set_test_mode"):
                prev_mode = getattr(input_ctrl, "test_mode", True)
                input_ctrl.set_test_mode(True)
                try:
                    if hasattr(input_ctrl, "press_key"):
                        input_ctrl.press_key(hotkey)
                finally:
                    input_ctrl.set_test_mode(prev_mode)
            executed_steps.extend(
                [
                    f"open_window('{hotkey}')",
                    f"select_category('{recipe.category}', norm={recipe.tab_norm_coords})",
                    f"select_recipe('{recipe.output_item}', norm={recipe.recipe_norm_coords})",
                    f"click_craft_all(norm={recipe.craft_all_norm_coords})",
                    "close_window('esc')",
                ]
            )
            # Sanal yurutme: gercek girdi gonderilmez, simule adimlar
            # test ortami icin basari sayilir.
            self.last_ui_crafting_ok = True
        return executed_steps

    def process_crafting_queue(
        self,
        input_ctrl: Optional[Any] = None,
        test_mode: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Kuyrukta bekleyen veya envanterde eşik değerine (`threshold`) ulaşmış tüm
        ham materyaller için Üretim / Substance Morphing işlemlerini çalıştırır,
        ham materyalleri düşer, üretilen ürünleri `crafted_inventory` içine ekler
        ve tamamlanan üretim görevlerinin listesini döndürür.

        Varsayilan test_mode=False'dur (F-0013): gercek modda girdi
        denetleyicisi yoksa oyuna hicbir sey gonderilmeden 'completed'
        yazilmaz; isler kuyrukta kalir ve bos liste doner.

        SIRA GARANTISI (S-0601): UI uretim sirasi ONCE calisir, envanter
        mutasyonu (ham dus, urun ekle) yalnizca UI basariliysa yapilir.
        UI basarisizsa (tiklama yetenegi eksik / girdi istisnasi) ham
        madde dusulmez, urun eklenmez, history'ye 'completed' yazilmaz;
        is yeniden kuyruga alinir.
        """
        completed_jobs: List[Dict[str, Any]] = []

        if not test_mode and input_ctrl is None:
            logger.error(
                "process_crafting_queue: gercek modda input_ctrl yok; "
                "hicbir uretim yapilmadi, %d is kuyrukta birakildi.",
                len(self.pending_queue),
            )
            return completed_jobs

        with self._lock:
            # Kuyrukta henüz yer almayan ama envanterde eşiği geçmiş materyalleri de yakala
            for mat_name, raw_qty in list(self.raw_inventory.items()):
                recipe = self._find_recipe(mat_name)
                if raw_qty >= recipe.threshold:
                    already_in_q = any(
                        j.get("raw_material") == mat_name and j.get("status") == "queued"
                        for j in self.pending_queue
                    )
                    if not already_in_q:
                        self.pending_queue.append(
                            {
                                "job_id": f"craft_{int(time.time() * 1000)}_{len(self.crafting_history) + 1}",
                                "raw_material": mat_name,
                                "output_item": recipe.output_item,
                                "category": recipe.category,
                                "threshold": recipe.threshold,
                                "available_raw_count": raw_qty,
                                "batches": max(1, raw_qty // max(1, recipe.batch_input_count)),
                                "window_hotkey": recipe.window_hotkey,
                                "status": "queued",
                                "queued_at": time.time(),
                            }
                        )

            jobs_to_run = list(self.pending_queue)
            self.pending_queue.clear()

        insufficient_jobs: List[Dict[str, Any]] = []
        deferred_jobs: List[Dict[str, Any]] = []
        for job in jobs_to_run:
            raw_mat = str(job["raw_material"])
            recipe = self._find_recipe(raw_mat)

            # 1) Plan: yalnizca oku, henuz envanteri degistirme (S-0601).
            with self._lock:
                current_raw = self.raw_inventory.get(raw_mat, 0)
                batch_cost = max(1, int(recipe.batch_input_count))
                if current_raw < batch_cost:
                    # Ham madde yetersizse is sessizce kaybolmaz (F-0085):
                    # yeniden kuyruga alinir ve loglanir.
                    logger.warning(
                        "AutoCrafting: '%s' icin ham madde yetersiz "
                        "(%d < %d); is yeniden kuyruga alindi.",
                        raw_mat,
                        current_raw,
                        batch_cost,
                    )
                    insufficient_jobs.append(job)
                    continue
                batches = max(1, current_raw // batch_cost)
                consumed_raw = batches * batch_cost
                produced_qty = batches * max(1, int(recipe.output_quantity))

            # 2) UI uretim sirasi ONCE: basarisizsa envantere dokunulmaz.
            ui_steps = self._execute_ui_crafting_sequence(
                recipe=recipe,
                input_ctrl=input_ctrl,
                test_mode=test_mode,
            )
            if not self.last_ui_crafting_ok:
                logger.error(
                    "AutoCrafting: '%s' uretimi UI sirasinda basarisiz "
                    "(%s); ham madde dusulmedi, urun eklenmedi, is "
                    "yeniden kuyruga alindi.",
                    raw_mat,
                    self.last_ui_crafting_error or "bilinmeyen hata",
                )
                deferred_jobs.append(job)
                continue

            # 3) UI basarili: envanter mutasyonu, kilit altinda yeniden
            # dogrulayarak (plan ile UI arasinda envanter degismis olabilir).
            with self._lock:
                current_raw = self.raw_inventory.get(raw_mat, 0)
                if current_raw < consumed_raw:
                    logger.warning(
                        "AutoCrafting: '%s' icin ham madde UI sirasinda "
                        "azaldi (%d < %d); is yeniden kuyruga alindi.",
                        raw_mat,
                        current_raw,
                        consumed_raw,
                    )
                    deferred_jobs.append(job)
                    continue
                self.raw_inventory[raw_mat] = max(0, current_raw - consumed_raw)
                self.crafted_inventory[recipe.output_item] = (
                    self.crafted_inventory.get(recipe.output_item, 0) + produced_qty
                )
                remaining_raw = self.raw_inventory[raw_mat]

            # 4) Degerleme: bilinmeyen ham/urun uydurma fiyatla
            # degerlenmez (S-0602); deger bilinmiyorsa 0 + bayrak.
            raw_unit_price = _known_unit_price(self.price_scanner, raw_mat)
            out_unit_price = _known_unit_price(self.price_scanner, recipe.output_item)
            unpriced: List[str] = []
            if raw_unit_price is None:
                unpriced.append(raw_mat)
            if out_unit_price is None:
                unpriced.append(recipe.output_item)
            valuation_known = not unpriced
            if unpriced:
                logger.warning(
                    "AutoCrafting: '%s' -> '%s' degerlemesi bilinmiyor "
                    "(fiyatsiz: %s); Kinah degerleri 0 yazildi, uydurma "
                    "fiyat kullanilmadi.",
                    raw_mat,
                    recipe.output_item,
                    ", ".join(unpriced),
                )
            raw_value = consumed_raw * (raw_unit_price or 0)
            crafted_value = produced_qty * (out_unit_price or 0)
            # Zarar eden uretim sifirlanarak gizlenmez (F-0255): negatif
            # katma deger oldugu gibi raporlanir.
            net_value_added = crafted_value - raw_value if valuation_known else 0

            result_record = {
                "job_id": job["job_id"],
                "raw_material": raw_mat,
                "consumed_raw_count": int(consumed_raw),
                "remaining_raw_count": int(remaining_raw),
                "output_item": recipe.output_item,
                "produced_quantity": int(produced_qty),
                "batches_executed": int(batches),
                "category": recipe.category,
                "window_hotkey": recipe.window_hotkey,
                "ui_steps": ui_steps,
                "raw_kinah_value": int(raw_value),
                "crafted_kinah_value": int(crafted_value),
                "net_kinah_value_added": int(net_value_added),
                "valuation_known": bool(valuation_known),
                "unpriced_materials": list(unpriced),
                "test_mode": bool(test_mode),
                "status": "completed",
                "completed_at": time.time(),
            }

            with self._lock:
                self.crafting_history.append(result_record)
            completed_jobs.append(result_record)

            logger.info(
                "[AutoCrafting] Tamamlandı: %dx %s -> %dx %s (Katma Değer: %s Kinah)",
                consumed_raw,
                raw_mat,
                produced_qty,
                recipe.output_item,
                format_kinah_tr(net_value_added),
            )

        if insufficient_jobs or deferred_jobs:
            with self._lock:
                self.pending_queue.extend(insufficient_jobs)
                self.pending_queue.extend(deferred_jobs)

        return completed_jobs


# ==============================================================================
# 3. SESSION PROFIT & KINAH SIMULATOR (HUD & MOBILE TELEMETRY)
# ==============================================================================
class SessionProfitSimulator:
    """
    Oturum boyunca toplanan tüm materyallerin ve üretilen ürünlerin gerçek zamanlı
    Kinah (Altın) değerini, saatlik kazanç hızını (`kinah_per_hour`) ve Türkçe
    özet metnini (`formatted_summary_tr`) Masaüstü HUD ve Mobil Panel için hesaplar.
    """

    def __init__(
        self,
        price_scanner: Optional[TradeBrokerPriceScanner] = None,
        crafting_queue: Optional[AutoCraftingMorphQueue] = None,
        broker_tax_rate: float = 0.0,
    ) -> None:
        self.price_scanner: TradeBrokerPriceScanner = price_scanner or TradeBrokerPriceScanner()
        self.crafting_queue: Optional[AutoCraftingMorphQueue] = crafting_queue
        self._lock = threading.Lock()
        self.session_start_ts: float = time.time()
        self.gathered_counts: Dict[str, int] = {}
        # Net deger hesabi icin opsiyonel komisyon orani (F-0430: artik
        # __init__ parametresi ve set_broker_tax_rate ile ayarlanabilir).
        self.broker_tax_rate: float = 0.0
        self.set_broker_tax_rate(broker_tax_rate)

    def set_broker_tax_rate(self, rate: float) -> float:
        """Broker komisyon oranini ayarlar; 0.0..1.0 araligi disinda ValueError."""
        try:
            value = float(rate)
        except (TypeError, ValueError):
            raise ValueError(f"broker_tax_rate sayisal olmali: {rate!r}") from None
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"broker_tax_rate 0.0..1.0 araliginda olmali: {value}")
        with self._lock:
            self.broker_tax_rate = value
        return self.broker_tax_rate

    def reset_session(self) -> None:
        """Oturum istatistiklerini ve başlangıç zamanını sıfırlar."""
        with self._lock:
            self.session_start_ts = time.time()
            self.gathered_counts.clear()
        # Uretim bonusu crafting_history'den geldigi icin sifirlanmazsa
        # eski bonus yeni oturum ozetlerine girmeye devam eder (F-0431).
        if self.crafting_queue is not None:
            with self.crafting_queue._lock:
                self.crafting_queue.crafting_history.clear()
                self.crafting_queue.crafted_inventory.clear()
                # S-0701: raw_inventory ve pending_queue da sifirlanir.
                # Sifirlanmazsa get_session_economy_summary counts bosken
                # raw_inventory'den senkronize edip sifirlanmis oturumu
                # eski materyallerle doldurur, bekleyen isler de yeni
                # oturuma sarkar; reset gercekte reset olmaz.
                self.crafting_queue.raw_inventory.clear()
                self.crafting_queue.pending_queue.clear()

    def record_gather(
        self,
        material_name: str,
        count: int = 1,
        forward_to_crafting: bool = True,
    ) -> int:
        """
        Toplanan materyali oturum sayacına kaydeder ve opsiyonel olarak
        `AutoCraftingMorphQueue` envanterine de iletir.
        """
        if not material_name or int(count) <= 0:
            return 0
        clean = str(material_name).strip()
        added = int(count)
        with self._lock:
            new_total = self.gathered_counts.get(clean, 0) + added
            self.gathered_counts[clean] = new_total

        if forward_to_crafting and self.crafting_queue is not None:
            self.crafting_queue.record_gathered_material(clean, added)

        return new_total

    def record_gathered_material(self, material_name: str, count: int = 1) -> int:
        """`record_gather` için uyumluluk eş anlamlısı."""
        return self.record_gather(material_name=material_name, count=count)

    def get_session_economy_summary(
        self,
        elapsed_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Masaüstü HUD ve Mobil Gösterge Paneli için gerçek zamanlı finansal telemetri üretir:
        - `net_kinah_earned`: Oturumda toplanan tüm materyallerin toplam net Kinah değeri
        - `kinah_per_hour`: Saatlik projeksiyon Kinah/Altın kazanç hızı
        - `formatted_summary_tr`: Örn. "Bu oturumda net 1.84M Kinah (Altın) değerinde materyal toplandı (Saatlik: 2.45M/saat)"
        - `material_breakdown`: Materyal bazlı adet, birim fiyat ve ara toplam Kinah sözlüğü

        elapsed_hours verilmezse (varsayilan None) gercek oturum suresi
        kullanilir; varsayilan 1.0 saat yanlis saatlik kazanc veriyordu (F-0086).
        """
        with self._lock:
            counts_snapshot = dict(self.gathered_counts)
            start_ts = self.session_start_ts

        # Eğer oturum sayacı boşsa ve bağlı crafting_queue içinde geçmiş/envanter varsa oradan da senkronize et
        if not counts_snapshot and self.crafting_queue is not None:
            with self.crafting_queue._lock:
                for mat_name, qty in self.crafting_queue.raw_inventory.items():
                    if qty > 0:
                        counts_snapshot[mat_name] = counts_snapshot.get(mat_name, 0) + int(qty)
                for hist in self.crafting_queue.crafting_history:
                    r_mat = str(hist.get("raw_material", ""))
                    c_qty = int(hist.get("consumed_raw_count", 0))
                    if r_mat and c_qty > 0:
                        counts_snapshot[r_mat] = counts_snapshot.get(r_mat, 0) + c_qty

        if elapsed_hours is None or float(elapsed_hours) <= 0.0:
            real_elapsed_sec = max(1.0, time.time() - start_ts)
            effective_hours = max(1.0 / 3600.0, real_elapsed_sec / 3600.0)
        else:
            effective_hours = max(1.0 / 3600.0, float(elapsed_hours))

        gross_kinah = 0
        total_items = 0
        material_breakdown: Dict[str, Dict[str, Any]] = {}
        unpriced_materials: Dict[str, int] = {}

        for mat_name, qty in sorted(counts_snapshot.items()):
            if qty <= 0:
                continue
            total_items += int(qty)
            # Bilinmeyen ad uydurma 12000 Kinah ile degerlenmez (S-0602):
            # brüte katilmaz, ayri raporlanir.
            unit_price = _known_unit_price(self.price_scanner, mat_name)
            if unit_price is None:
                unpriced_materials[mat_name] = int(qty)
                continue
            subtotal = int(qty) * int(unit_price)
            gross_kinah += subtotal
            material_breakdown[mat_name] = {
                "material": mat_name,
                "quantity": int(qty),
                "unit_price": int(unit_price),
                "subtotal_kinah": int(subtotal),
                "formatted_subtotal": format_kinah_tr(subtotal),
            }
        if unpriced_materials:
            logger.warning(
                "get_session_economy_summary: fiyati bilinmeyen materyal(ler) "
                "brüt kazanca katilmadi (uydurma fiyat yok): %s",
                ", ".join(sorted(unpriced_materials)),
            )

        # Crafting katma değeri varsa ekle
        crafted_bonus_kinah = 0
        if self.crafting_queue is not None:
            with self.crafting_queue._lock:
                for job_rec in self.crafting_queue.crafting_history:
                    crafted_bonus_kinah += int(job_rec.get("net_kinah_value_added", 0))

        total_gross_with_craft = gross_kinah + crafted_bonus_kinah
        tax_multiplier = max(0.0, min(1.0, 1.0 - float(self.broker_tax_rate)))
        net_kinah_earned = int(round(total_gross_with_craft * tax_multiplier))
        kinah_per_hour = int(round(net_kinah_earned / effective_hours))

        # Her materyalin toplam içindeki payını (%) hesapla
        denom = float(max(1, gross_kinah))
        for mat_info in material_breakdown.values():
            mat_info["share_pct"] = round((mat_info["subtotal_kinah"] / denom) * 100.0, 1)

        formatted_net = format_kinah_tr(net_kinah_earned)
        formatted_hourly = format_kinah_tr(kinah_per_hour)
        formatted_summary_tr = (
            f"Bu oturumda net {formatted_net} Kinah (Altın) değerinde materyal toplandı "
            f"(Saatlik: {formatted_hourly}/saat)"
        )

        return {
            "net_kinah_earned": int(net_kinah_earned),
            "gross_kinah_earned": int(gross_kinah),
            "crafted_bonus_kinah": int(crafted_bonus_kinah),
            "kinah_per_hour": int(kinah_per_hour),
            "elapsed_hours": round(effective_hours, 4),
            "total_items_gathered": int(total_items),
            "formatted_net_kinah": formatted_net,
            "formatted_kinah_per_hour": formatted_hourly,
            "formatted_summary_tr": formatted_summary_tr,
            "material_breakdown": material_breakdown,
            "unpriced_materials": dict(unpriced_materials),
        }

    def get_economy_broker_health_summary(self) -> dict:
        # Gercek denetim: fiyat tablosu ve tarayici ornegi calisiyor mu (F-0256).
        try:
            ok = bool(DEFAULT_BROKER_PRICES) and bool(self.price_scanner.get_price("Orichalcum Ore", 0) > 0)
            return {"economy_broker_ready": bool(ok)}
        except Exception:
            return {"economy_broker_ready": False}

    def get_economy_broker_ready_flag(self) -> dict:
        return {"module": "economy_broker", "ready": bool(validate_economy_broker_contract())}

    def get_economy_broker_version_info(self) -> dict:
        return {"module": "economy_broker", "version": __version__, "ready": bool(validate_economy_broker_contract())}

    def calculate_net_hourly_kinah(
        self,
        gathered_counts: Dict[str, int],
        consumables_used: Dict[str, int],
        elapsed_hours: float = 1.0,
    ) -> Dict[str, Any]:
        """Seçili materyallere göre iksir/parşömen maliyetini düşerek saatlik net Kinah kârını hesaplar."""
        return calculate_net_hourly_kinah(
            gathered_counts=gathered_counts,
            consumables_used=consumables_used,
            elapsed_hours=elapsed_hours,
            price_scanner=self,
        )

    def get_economy_broker_diagnostic_snapshot(self) -> dict:
        ready = bool(validate_economy_broker_contract())
        return {"module": "economy_broker", "status": "ok" if ready else "degraded", "ready": ready}


def run_economy_cycle(
    scanner: Optional[TradeBrokerPriceScanner] = None,
    route_manager: Optional[Any] = None,
    crafting_queue: Optional[AutoCraftingMorphQueue] = None,
    input_ctrl: Optional[Any] = None,
    frame: Optional[np.ndarray] = None,
    region: Optional[str] = None,
    test_mode: bool = False,
) -> Dict[str, Any]:
    """
    Ekonomi zincirinin tek üretim giriş noktası (F-0088): pazar tarama ->
    en kârlı rota seçimi -> üretim kuyruğu, doğru sırada tek çağrıdan
    çalışır. Bot döngüsüne (BotWorker / main) bağlanması gerekir; bu
    dosya kendi başına çağırmaz. Çağrı noktası core/state_machine.py
    tarafında olduğu ve bu dilim yalnızca core/economy_broker.py
    kapsadığı için F-0088 bu dilimde not_fixed bırakıldı — bağlama
    işi ayrı dilimin işidir; bu fonksiyon o bağlamayı tek satıra indirir.
    """
    result: Dict[str, Any] = {
        "scanned_prices": None,
        "route": None,
        "crafted_jobs": [],
    }
    if frame is not None and scanner is not None:
        result["scanned_prices"] = scanner.scan_broker_window(frame)
    if route_manager is not None and scanner is not None:
        result["route"] = scanner.select_most_profitable_route(
            route_manager, region=region, activate=True
        )
    if crafting_queue is not None:
        result["crafted_jobs"] = crafting_queue.process_crafting_queue(
            input_ctrl=input_ctrl, test_mode=test_mode
        )
    return result


def get_economy_broker_capabilities() -> list:
    return ["economy_broker", "market_valuation", "diagnostic_snapshot", "run_economy_cycle"]


def validate_economy_broker_contract() -> bool:
    """
    Gercek sozlesme denetimi (F-0256): sabit True degil. Fiyat tablosu
    dolu mu, format fonksiyonu dokumandaki ornegi veriyor mu, tarayici
    gecici fiyat dosyasıyla orneklenip fiyat okuyabiliyor mu, receteler
    var mi diye bakar; herhangi biri bozulursa False doner.
    """
    try:
        if not DEFAULT_BROKER_PRICES or not DEFAULT_CRAFTING_RECIPES:
            return False
        if format_kinah_tr(42500) != "42.5K":
            return False
        if not get_economy_broker_capabilities():
            return False
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            scanner = TradeBrokerPriceScanner(
                price_db_path=os.path.join(tmp_dir, "prices.json"),
                auto_save=False,
            )
            if scanner.get_price("Orichalcum Ore", 0) <= 0:
                return False
            if scanner.get_price("Pure Odyle", 0) == scanner.get_price("Odyle", 0):
                return False
        return True
    except Exception:
        return False


def get_economy_broker_baseline_metrics() -> dict:
    return {"module": "economy_broker", "contract_valid": validate_economy_broker_contract(), "capability_count": len(get_economy_broker_capabilities())}


def get_economy_broker_readiness_status() -> dict:
    return {"module": "economy_broker", "ready": validate_economy_broker_contract(), "metrics": get_economy_broker_baseline_metrics()}


def get_economy_broker_audit_record() -> dict:
    return {"module": "economy_broker", "audit_passed": validate_economy_broker_contract(), "readiness": get_economy_broker_readiness_status()}


def get_economy_broker_operational_digest() -> dict:
    return {"module": "economy_broker", "operational": validate_economy_broker_contract(), "audit": get_economy_broker_audit_record()}


def get_economy_broker_telemetry_payload() -> dict:
    return {"module": "economy_broker", "payload": get_economy_broker_operational_digest()}


def get_economy_broker_integrity_seal() -> dict:
    return {"module": "economy_broker", "sealed": validate_economy_broker_contract(), "telemetry": get_economy_broker_telemetry_payload()}


def get_economy_broker_compliance_certificate() -> dict:
    return {"module": "economy_broker", "certified": validate_economy_broker_contract(), "seal": get_economy_broker_integrity_seal()}


def get_economy_broker_readiness_probe() -> dict:
    return {"module": "economy_broker", "ready": validate_economy_broker_contract(), "certificate": get_economy_broker_compliance_certificate()}


def get_economy_broker_audit_summary() -> dict:
    return {"module": "economy_broker", "audit_passed": validate_economy_broker_contract(), "probe": get_economy_broker_readiness_probe()}

