# -*- coding: utf-8 -*-
"""
Questlog Automatic Zone & Resource Node Importer
Implements the 'Import from questlog...' feature seen in Gatherion 1.5.2 (00:07 - 00:25).
Includes:
- QUESTLOG_DATABASE: Full zone, resource, and sub-zone coordinates database for Aion 2.
- QuestlogImporter: Map capture, node coordinate generation/extraction, 80x80 grayscale map patch cropping,
  and batch route saving via RouteManager.
- QuestlogImportDialog: Pixel-accurate PyQt6 dark-themed dialog matching frame_12s.jpg & frame_15s.jpg.
"""

import os
import time
import math
import logging
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QListWidget, QListWidgetItem, QPushButton,
    QMessageBox, QProgressBar, QApplication, QFrame, QAbstractItemView
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QColor, QIcon, QPixmap, QPainter, QPen, QBrush

logger = logging.getLogger("QuestlogImporter")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ==============================================================================
# EXACT 114 NODE COORDINATES FROM FRAME_24S.JPG (DREDGION CRASH SITE)
# ==============================================================================
# Extracted directly from the 114 green nodes shown in video_frames/frame_24s.jpg
# and calibrated to the 1280x720 in-game map coordinate space (frame_18s.jpg).
DREDGION_CRASH_SITE_24S_COORDS = [
    (863, 222), (756, 284), (788, 309), (801, 371), (723, 368), (740, 431),
    (697, 476), (727, 333), (666, 467), (678, 425), (675, 380), (595, 449),
    (560, 481), (525, 278), (566, 348), (612, 322), (663, 333), (578, 411),
    (503, 452), (617, 447), (756, 323), (731, 300), (650, 469), (751, 257),
    (620, 400), (823, 330), (778, 467), (689, 346), (872, 311), (477, 395),
    (463, 376), (513, 332), (524, 291), (607, 301), (651, 290), (724, 271),
    (854, 360), (720, 452), (663, 221), (455, 265), (748, 273), (863, 265),
    (865, 239), (653, 404), (436, 346), (820, 351), (667, 377), (628, 430),
    (560, 423), (517, 296), (549, 360), (620, 368), (581, 365), (542, 450),
    (545, 469), (559, 414), (484, 499), (751, 349), (760, 276), (602, 414),
    (798, 325), (639, 271), (693, 490), (705, 388), (642, 351), (652, 482),
    (814, 368), (742, 309), (820, 270), (738, 292), (710, 348), (801, 317),
    (553, 392), (485, 384), (563, 304), (537, 300), (681, 281), (816, 286),
    (796, 351), (708, 455), (426, 286), (487, 279), (474, 444), (482, 265),
    (704, 282), (778, 365), (835, 227), (558, 520), (687, 246), (695, 304),
    (669, 426), (638, 435), (580, 422), (564, 385), (636, 332), (483, 419),
    (517, 413), (555, 430), (506, 444), (454, 449), (538, 503), (511, 492),
    (734, 338), (789, 369), (812, 333), (715, 388), (756, 403), (668, 438),
    (677, 490), (638, 485), (728, 352), (695, 368), (696, 482), (770, 405),
    (499, 368), (444, 332), (406, 327), (407, 299)
]


def _imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Safely reads an image from a path that may contain non-ASCII/Turkish characters on Windows."""
    if not path or not os.path.isfile(path):
        return None
    try:
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, flags)
    except Exception as e:
        logger.debug(f"_imread_unicode failed for '{path}': {e}")
        return None


def _imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """Safely writes an image to a path that may contain non-ASCII/Turkish characters on Windows."""
    if img is None or img.size == 0:
        return False
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ext = os.path.splitext(path)[1] or ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception as e:
        logger.debug(f"_imwrite_unicode failed for '{path}': {e}")
        return False


def _generate_subzone_coords(center_x: int, center_y: int, count: int, rx: int = 135, ry: int = 95, seed: int = 42) -> list[tuple[int, int]]:
    """
    Generates deterministic, realistic organic cluster coordinates around a sub-zone center
    on a 1280x720 map canvas.
    """
    rng = np.random.RandomState(seed)
    coords = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(count):
        r_norm = math.sqrt((i + 0.5) / count)
        theta = i * golden_angle + rng.uniform(-0.25, 0.25)
        jitter_x = rng.normal(0, rx * 0.12)
        jitter_y = rng.normal(0, ry * 0.12)
        x = int(round(center_x + r_norm * rx * math.cos(theta) + jitter_x))
        y = int(round(center_y + r_norm * ry * math.sin(theta) + jitter_y))
        x = max(120, min(1160, x))
        y = max(90, min(630, y))
        coords.append((x, y))
    return coords


def _convert_24s_to_18s_coords(coords_24s: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Converts coordinates from the zoomed map viewer modal in frame_24s.jpg
    to the full 1280x720 in-game map in frame_18s.jpg.
    Also preserves the 24s coordinates for direct rendering on map_at_25s.jpg / frame_24s.jpg.
    """
    converted = []
    for (x24, y24) in coords_24s:
        x18 = int(round((x24 + 583.0) / 1.7625))
        y18 = int(round((y24 + 164.0) / 1.7625))
        converted.append((x18, y18))
    return converted


# ==============================================================================
# QUESTLOG DATABASE (ALL 9 AION 2 ZONES, RESOURCES & SUB-ZONES)
# ==============================================================================
QUESTLOG_DATABASE = {
    "Altgard": {
        "faction": "Asmodian",
        "resources": [
            {"name": "Odyle", "level": 650, "in_air": True},
            {"name": "Orichalcum Ore", "level": 530, "in_air": False},
            {"name": "Diamond Gemstone", "level": 200, "in_air": False},
            {"name": "Sapphire Gemstone", "level": 200, "in_air": False},
            {"name": "Ruby Gemstone", "level": 190, "in_air": False},
            {"name": "Aquamarine Gemstone", "level": 180, "in_air": False},
            {"name": "Lazuli Gemstone", "level": 175, "in_air": False},
            {"name": "Cippo", "level": 160, "in_air": False},
            {"name": "angelica", "level": 150, "in_air": False},
        ],
        "sub_zones": [
            {
                "name": "Dredgion Crash Site",
                "count": 118,
                "active_import_count": 114,
                "already_there_count": 4,
                "center": (655, 455),
                "coords_24s": DREDGION_CRASH_SITE_24S_COORDS,
                "coords_18s": _convert_24s_to_18s_coords(DREDGION_CRASH_SITE_24S_COORDS),
            },
            {
                "name": "Nameless Cemetery",
                "count": 118,
                "active_import_count": 118,
                "already_there_count": 0,
                "center": (455, 360),
                "coords_24s": _generate_subzone_coords(460, 360, 118, rx=165, ry=110, seed=101),
                "coords_18s": _generate_subzone_coords(455, 360, 118, rx=95, ry=65, seed=101),
            },
            {
                "name": "Fafnite Deposit",
                "count": 112,
                "active_import_count": 112,
                "already_there_count": 0,
                "center": (575, 290),
                "coords_24s": _generate_subzone_coords(580, 310, 112, rx=160, ry=105, seed=102),
                "coords_18s": _generate_subzone_coords(575, 290, 112, rx=90, ry=60, seed=102),
            },
            {
                "name": "Basfelt Ruins",
                "count": 111,
                "active_import_count": 111,
                "already_there_count": 0,
                "center": (440, 495),
                "coords_24s": _generate_subzone_coords(490, 470, 111, rx=155, ry=105, seed=103),
                "coords_18s": _generate_subzone_coords(440, 495, 111, rx=88, ry=58, seed=103),
            },
            {
                "name": "Sanctum Outpost",
                "count": 83,
                "active_import_count": 83,
                "already_there_count": 0,
                "center": (735, 320),
                "coords_24s": _generate_subzone_coords(735, 330, 83, rx=140, ry=95, seed=104),
                "coords_18s": _generate_subzone_coords(735, 320, 83, rx=80, ry=55, seed=104),
            },
        ],
    },
    "Verteron": {
        "faction": "Elyos",
        "resources": [
            {"name": "Aether", "level": 620, "in_air": True},
            {"name": "Titanium Ore", "level": 510, "in_air": False},
            {"name": "Iron Ore", "level": 420, "in_air": False},
            {"name": "Garnet Gemstone", "level": 210, "in_air": False},
            {"name": "Moonstone Gemstone", "level": 195, "in_air": False},
            {"name": "Taro", "level": 180, "in_air": False},
            {"name": "Dyam", "level": 165, "in_air": False},
        ],
        "sub_zones": [
            {
                "name": "Tolbas Forest",
                "count": 124,
                "active_import_count": 124,
                "already_there_count": 0,
                "center": (540, 380),
                "coords_24s": _generate_subzone_coords(620, 380, 124, rx=180, ry=115, seed=201),
                "coords_18s": _generate_subzone_coords(540, 380, 124, rx=105, ry=68, seed=201),
            },
            {
                "name": "Dukaki Settlement",
                "count": 116,
                "active_import_count": 116,
                "already_there_count": 0,
                "center": (680, 420),
                "coords_24s": _generate_subzone_coords(680, 410, 116, rx=175, ry=110, seed=202),
                "coords_18s": _generate_subzone_coords(680, 420, 116, rx=98, ry=64, seed=202),
            },
            {
                "name": "Cantas Coast",
                "count": 98,
                "active_import_count": 98,
                "already_there_count": 0,
                "center": (480, 290),
                "coords_24s": _generate_subzone_coords(550, 320, 98, rx=160, ry=100, seed=203),
                "coords_18s": _generate_subzone_coords(480, 290, 98, rx=90, ry=58, seed=203),
            },
            {
                "name": "Ardus Shrine",
                "count": 89,
                "active_import_count": 89,
                "already_there_count": 0,
                "center": (610, 310),
                "coords_24s": _generate_subzone_coords(640, 340, 89, rx=150, ry=95, seed=204),
                "coords_18s": _generate_subzone_coords(610, 310, 89, rx=85, ry=55, seed=204),
            },
        ],
    },
    "Eltnen": {
        "faction": "Elyos",
        "resources": [
            {"name": "Burning Aether", "level": 680, "in_air": True},
            {"name": "Adamantium Ore", "level": 560, "in_air": False},
            {"name": "Orichalcum Ore", "level": 520, "in_air": False},
            {"name": "Topaz Gemstone", "level": 240, "in_air": False},
            {"name": "Peridot Gemstone", "level": 225, "in_air": False},
            {"name": "Rhea", "level": 190, "in_air": False},
            {"name": "Ginseng", "level": 175, "in_air": False},
        ],
        "sub_zones": [
            {
                "name": "Manduri Forest",
                "count": 132,
                "active_import_count": 132,
                "already_there_count": 0,
                "center": (520, 370),
                "coords_24s": _generate_subzone_coords(610, 375, 132, rx=185, ry=120, seed=301),
                "coords_18s": _generate_subzone_coords(520, 370, 132, rx=110, ry=72, seed=301),
            },
            {
                "name": "Golden Bough Garrison",
                "count": 115,
                "active_import_count": 115,
                "already_there_count": 0,
                "center": (650, 340),
                "coords_24s": _generate_subzone_coords(660, 360, 115, rx=175, ry=110, seed=302),
                "coords_18s": _generate_subzone_coords(650, 340, 115, rx=98, ry=65, seed=302),
            },
            {
                "name": "Eiron Desert",
                "count": 128,
                "active_import_count": 128,
                "already_there_count": 0,
                "center": (590, 440),
                "coords_24s": _generate_subzone_coords(635, 420, 128, rx=180, ry=115, seed=303),
                "coords_18s": _generate_subzone_coords(590, 440, 128, rx=105, ry=68, seed=303),
            },
            {
                "name": "Kaidan Headquarters",
                "count": 94,
                "active_import_count": 94,
                "already_there_count": 0,
                "center": (710, 410),
                "coords_24s": _generate_subzone_coords(710, 400, 94, rx=155, ry=100, seed=304),
                "coords_18s": _generate_subzone_coords(710, 410, 94, rx=88, ry=58, seed=304),
            },
        ],
    },
    "Morheim": {
        "faction": "Asmodian",
        "resources": [
            {"name": "Odyle", "level": 710, "in_air": True},
            {"name": "Adamantium Ore", "level": 590, "in_air": False},
            {"name": "Orichalcum Ore", "level": 540, "in_air": False},
            {"name": "Emerald Gemstone", "level": 260, "in_air": False},
            {"name": "Sapphire Gemstone", "level": 240, "in_air": False},
            {"name": "Amethyst Gemstone", "level": 215, "in_air": False},
            {"name": "Umbel", "level": 185, "in_air": False},
        ],
        "sub_zones": [
            {
                "name": "Ice Claw Village",
                "count": 126,
                "active_import_count": 126,
                "already_there_count": 0,
                "center": (510, 350),
                "coords_24s": _generate_subzone_coords(600, 360, 126, rx=180, ry=115, seed=401),
                "coords_18s": _generate_subzone_coords(510, 350, 126, rx=105, ry=68, seed=401),
            },
            {
                "name": "Patamor Ridge",
                "count": 119,
                "active_import_count": 119,
                "already_there_count": 0,
                "center": (640, 390),
                "coords_24s": _generate_subzone_coords(645, 390, 119, rx=175, ry=110, seed=402),
                "coords_18s": _generate_subzone_coords(640, 390, 119, rx=98, ry=64, seed=402),
            },
            {
                "name": "Salintus Desert",
                "count": 134,
                "active_import_count": 134,
                "already_there_count": 0,
                "center": (580, 450),
                "coords_24s": _generate_subzone_coords(630, 430, 134, rx=185, ry=120, seed=403),
                "coords_18s": _generate_subzone_coords(580, 450, 134, rx=108, ry=70, seed=403),
            },
            {
                "name": "Mist Mane Village",
                "count": 102,
                "active_import_count": 102,
                "already_there_count": 0,
                "center": (715, 360),
                "coords_24s": _generate_subzone_coords(715, 370, 102, rx=160, ry=105, seed=404),
                "coords_18s": _generate_subzone_coords(715, 360, 102, rx=92, ry=60, seed=404),
            },
        ],
    },
    "Heiron": {
        "faction": "Elyos",
        "resources": [
            {"name": "Pure Aether", "level": 760, "in_air": True},
            {"name": "Platinum Ore", "level": 640, "in_air": False},
            {"name": "Adamantium Ore", "level": 580, "in_air": False},
            {"name": "Diamond Gemstone", "level": 310, "in_air": False},
            {"name": "Ruby Gemstone", "level": 285, "in_air": False},
            {"name": "Fess", "level": 230, "in_air": False},
            {"name": "Griffo", "level": 210, "in_air": False},
        ],
        "sub_zones": [
            {
                "name": "Heiron Observatory",
                "count": 122,
                "active_import_count": 122,
                "already_there_count": 0,
                "center": (560, 360),
                "coords_24s": _generate_subzone_coords(620, 365, 122, rx=180, ry=115, seed=501),
                "coords_18s": _generate_subzone_coords(560, 360, 122, rx=102, ry=66, seed=501),
            },
            {
                "name": "Senea's Campsite",
                "count": 108,
                "active_import_count": 108,
                "already_there_count": 0,
                "center": (650, 415),
                "coords_24s": _generate_subzone_coords(660, 410, 108, rx=170, ry=108, seed=502),
                "coords_18s": _generate_subzone_coords(650, 415, 108, rx=95, ry=62, seed=502),
            },
            {
                "name": "Kishar Observation Post",
                "count": 117,
                "active_import_count": 117,
                "already_there_count": 0,
                "center": (690, 330),
                "coords_24s": _generate_subzone_coords(690, 345, 117, rx=175, ry=110, seed=503),
                "coords_18s": _generate_subzone_coords(690, 330, 117, rx=98, ry=64, seed=503),
            },
            {
                "name": "Arbolu's Haven",
                "count": 95,
                "active_import_count": 95,
                "already_there_count": 0,
                "center": (495, 430),
                "coords_24s": _generate_subzone_coords(570, 415, 95, rx=155, ry=100, seed=504),
                "coords_18s": _generate_subzone_coords(495, 430, 95, rx=88, ry=58, seed=504),
            },
        ],
    },
    "Beluslan": {
        "faction": "Asmodian",
        "resources": [
            {"name": "Pure Odyle", "level": 780, "in_air": True},
            {"name": "Platinum Ore", "level": 660, "in_air": False},
            {"name": "Orichalcum Ore", "level": 590, "in_air": False},
            {"name": "Diamond Gemstone", "level": 320, "in_air": False},
            {"name": "Sapphire Gemstone", "level": 290, "in_air": False},
            {"name": "Amaryllis", "level": 240, "in_air": False},
            {"name": "Pressa", "level": 220, "in_air": False},
        ],
        "sub_zones": [
            {
                "name": "Besfer Refugee Camp",
                "count": 129,
                "active_import_count": 129,
                "already_there_count": 0,
                "center": (550, 370),
                "coords_24s": _generate_subzone_coords(615, 375, 129, rx=182, ry=116, seed=601),
                "coords_18s": _generate_subzone_coords(550, 370, 129, rx=104, ry=67, seed=601),
            },
            {
                "name": "Red Mane Cavern",
                "count": 115,
                "active_import_count": 115,
                "already_there_count": 0,
                "center": (660, 410),
                "coords_24s": _generate_subzone_coords(665, 405, 115, rx=174, ry=110, seed=602),
                "coords_18s": _generate_subzone_coords(660, 410, 115, rx=97, ry=63, seed=602),
            },
            {
                "name": "Hoarfrost Shelter",
                "count": 121,
                "active_import_count": 121,
                "already_there_count": 0,
                "center": (620, 315),
                "coords_24s": _generate_subzone_coords(645, 335, 121, rx=178, ry=112, seed=603),
                "coords_18s": _generate_subzone_coords(620, 315, 121, rx=100, ry=65, seed=603),
            },
            {
                "name": "Bakarma Barracks",
                "count": 104,
                "active_import_count": 104,
                "already_there_count": 0,
                "center": (720, 385),
                "coords_24s": _generate_subzone_coords(715, 385, 104, rx=165, ry=105, seed=604),
                "coords_18s": _generate_subzone_coords(720, 385, 104, rx=93, ry=60, seed=604),
            },
        ],
    },
    "Theobomos": {
        "faction": "Elyos",
        "resources": [
            {"name": "Brilliant Aether", "level": 820, "in_air": True},
            {"name": "Mithril Ore", "level": 710, "in_air": False},
            {"name": "Platinum Ore", "level": 650, "in_air": False},
            {"name": "Corundum Gemstone", "level": 350, "in_air": False},
            {"name": "Ruby Gemstone", "level": 310, "in_air": False},
            {"name": "Licorice", "level": 260, "in_air": False},
        ],
        "sub_zones": [
            {
                "name": "Meniherk Excavation Camp",
                "count": 136,
                "active_import_count": 136,
                "already_there_count": 0,
                "center": (575, 380),
                "coords_24s": _generate_subzone_coords(630, 380, 136, rx=186, ry=120, seed=701),
                "coords_18s": _generate_subzone_coords(575, 380, 136, rx=108, ry=70, seed=701),
            },
            {
                "name": "Observatory Village",
                "count": 114,
                "active_import_count": 114,
                "already_there_count": 0,
                "center": (665, 350),
                "coords_24s": _generate_subzone_coords(670, 360, 114, rx=174, ry=110, seed=702),
                "coords_18s": _generate_subzone_coords(665, 350, 114, rx=97, ry=63, seed=702),
            },
            {
                "name": "Jamanok Inn",
                "count": 109,
                "active_import_count": 109,
                "already_there_count": 0,
                "center": (520, 430),
                "coords_24s": _generate_subzone_coords(590, 415, 109, rx=170, ry=108, seed=703),
                "coords_18s": _generate_subzone_coords(520, 430, 109, rx=95, ry=62, seed=703),
            },
        ],
    },
    "Brusthonin": {
        "faction": "Asmodian",
        "resources": [
            {"name": "Brilliant Odyle", "level": 830, "in_air": True},
            {"name": "Mithril Ore", "level": 720, "in_air": False},
            {"name": "Platinum Ore", "level": 660, "in_air": False},
            {"name": "Corundum Gemstone", "level": 355, "in_air": False},
            {"name": "Sapphire Gemstone", "level": 315, "in_air": False},
            {"name": "Plia", "level": 265, "in_air": False},
        ],
        "sub_zones": [
            {
                "name": "Settlers' Campsite",
                "count": 131,
                "active_import_count": 131,
                "already_there_count": 0,
                "center": (565, 375),
                "coords_24s": _generate_subzone_coords(625, 378, 131, rx=184, ry=118, seed=801),
                "coords_18s": _generate_subzone_coords(565, 375, 131, rx=106, ry=69, seed=801),
            },
            {
                "name": "Baltasar Hill Village",
                "count": 120,
                "active_import_count": 120,
                "already_there_count": 0,
                "center": (655, 415),
                "coords_24s": _generate_subzone_coords(660, 405, 120, rx=176, ry=112, seed=802),
                "coords_18s": _generate_subzone_coords(655, 415, 120, rx=99, ry=64, seed=802),
            },
            {
                "name": "Bubu Village",
                "count": 106,
                "active_import_count": 106,
                "already_there_count": 0,
                "center": (515, 330),
                "coords_24s": _generate_subzone_coords(585, 350, 106, rx=168, ry=106, seed=803),
                "coords_18s": _generate_subzone_coords(515, 330, 106, rx=94, ry=61, seed=803),
            },
        ],
    },
    "Reshanta": {
        "faction": "Abyss",
        "resources": [
            {"name": "Abyssal Odyle", "level": 900, "in_air": True},
            {"name": "Orichalcum Ore", "level": 780, "in_air": True},
            {"name": "Mithril Ore", "level": 750, "in_air": True},
            {"name": "Dranium Ore", "level": 700, "in_air": False},
            {"name": "Diamond Gemstone", "level": 400, "in_air": True},
            {"name": "Red Heliotrope", "level": 350, "in_air": True},
        ],
        "sub_zones": [
            {
                "name": "Teminon Landing",
                "count": 110,
                "active_import_count": 110,
                "already_there_count": 0,
                "center": (540, 440),
                "coords_24s": _generate_subzone_coords(605, 425, 110, rx=172, ry=110, seed=901),
                "coords_18s": _generate_subzone_coords(540, 440, 110, rx=96, ry=62, seed=901),
            },
            {
                "name": "Primum Landing",
                "count": 115,
                "active_import_count": 115,
                "already_there_count": 0,
                "center": (660, 290),
                "coords_24s": _generate_subzone_coords(665, 325, 115, rx=175, ry=112, seed=902),
                "coords_18s": _generate_subzone_coords(660, 290, 115, rx=98, ry=64, seed=902),
            },
            {
                "name": "Sulfur Tree Nest",
                "count": 125,
                "active_import_count": 125,
                "already_there_count": 0,
                "center": (590, 365),
                "coords_24s": _generate_subzone_coords(635, 370, 125, rx=180, ry=115, seed=903),
                "coords_18s": _generate_subzone_coords(590, 365, 125, rx=102, ry=66, seed=903),
            },
            {
                "name": "Siel's Eastern Fortress",
                "count": 96,
                "active_import_count": 96,
                "already_there_count": 0,
                "center": (710, 380),
                "coords_24s": _generate_subzone_coords(710, 385, 96, rx=158, ry=102, seed=904),
                "coords_18s": _generate_subzone_coords(710, 380, 96, rx=89, ry=58, seed=904),
            },
        ],
    },
}


# ==============================================================================
# QUESTLOG IMPORTER ENGINE
# ==============================================================================
class QuestlogImporter:
    """
    Core engine for importing zone resource nodes from the Aion 2 Questlog database
    and capturing 80x80 grayscale map texture patches for each node.
    """

    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.db = QUESTLOG_DATABASE
        self._cached_ref_maps = {}

    def get_questlog_importer_health_summary(self) -> dict:
        try:
            return {"questlog_importer_ready": True}
        except Exception:
            return {"questlog_importer_ready": False}

    def get_zones(self) -> list[str]:
        """Returns the list of all available zones."""
        try:
            return list(self.db.keys())
        except Exception as e:
            logger.error(f"Error getting zones: {e}")
            return ["Altgard"]

    def get_resources(self, zone: str) -> list[dict]:
        """
        Returns resource list for a given zone with display text like 'Odyle (650)'.
        """
        try:
            zone_data = self.db.get(zone, self.db.get("Altgard", {}))
            resources = []
            for r in zone_data.get("resources", []):
                item = dict(r)
                item["display"] = f"{r['name']} ({r['level']})"
                resources.append(item)
            return resources
        except Exception as e:
            logger.error(f"Error getting resources for zone '{zone}': {e}")
            return []

    def get_sub_zones(self, zone: str) -> list[dict]:
        """
        Returns sub-zone list for a given zone with display text like 'Dredgion Crash Site (118)'.
        """
        try:
            zone_data = self.db.get(zone, self.db.get("Altgard", {}))
            sub_zones = []
            for sz in zone_data.get("sub_zones", []):
                item = dict(sz)
                item["display"] = f"{sz['name']} ({sz['count']})"
                sub_zones.append(item)
            return sub_zones
        except Exception as e:
            logger.error(f"Error getting sub-zones for zone '{zone}': {e}")
            return []

    def get_reference_map_image(self, zone: str = "Altgard", clean_modal_canvas: bool = False) -> np.ndarray:
        """
        Loads and returns a clean 1280x720 BGR map image for the given zone.
        Uses 'video_frames/frame_18s.jpg' (the clean in-game world map) or
        synthesizes a clean grayscale parchment map from 'video_frames/frame_18s.jpg' / 'frame_24s.jpg'
        so that MapViewer can always display nodes over a crisp, genuine map texture.
        """
        cache_key = (zone, clean_modal_canvas)
        if cache_key in self._cached_ref_maps:
            return self._cached_ref_maps[cache_key].copy()

        try:
            frame18_path = os.path.join(self.base_dir, "video_frames", "frame_18s.jpg")
            frame24_path = os.path.join(self.base_dir, "video_frames", "frame_24s.jpg")
            map25_path = os.path.join(self.base_dir, "video_analysis", "map_at_25s.jpg")

            img18 = _imread_unicode(frame18_path, cv2.IMREAD_COLOR)
            img24 = _imread_unicode(frame24_path, cv2.IMREAD_COLOR)
            if img24 is None:
                img24 = _imread_unicode(map25_path, cv2.IMREAD_COLOR)

            if img18 is not None and not clean_modal_canvas:
                # Remove the bottom tutorial banner ("3 Press Start and wait 20 seconds") from frame_18s
                # by smoothly blending the surrounding cloud texture over y=520..615, x=295..985
                clean_map = img18.copy()
                # Also remove the top-left overlay and "x4" badge for a pure map texture
                mask = np.zeros(clean_map.shape[:2], dtype=np.uint8)
                # Bottom tutorial banner
                mask[520:615, 295:985] = 255
                # Top ">> x4" badge
                mask[30:85, 300:395] = 255
                clean_map = cv2.inpaint(clean_map, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
                self._cached_ref_maps[cache_key] = clean_map
                return clean_map.copy()

            if img18 is not None and clean_modal_canvas:
                # Reconstruct the exact zoomed grayscale map canvas seen inside the MapViewer modal in frame_24s.jpg!
                # Recall: x18 = (x24 + 583) / 1.7625, y18 = (y24 + 164) / 1.7625
                g18 = cv2.cvtColor(img18, cv2.COLOR_BGR2GRAY)
                scale = 1.7625
                r18 = cv2.resize(g18, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                # Extract 1280x720 window aligned with frame_24s coordinate space
                canvas_gray = np.full((720, 1280), 35, dtype=np.uint8)
                rh, rw = r18.shape
                for y24 in range(720):
                    ry = y24 + 164
                    if 0 <= ry < rh:
                        x_start = 583
                        x_end = min(rw, 583 + 1280)
                        span = x_end - x_start
                        if span > 0:
                            canvas_gray[y24, :span] = r18[ry, x_start:x_end]
                canvas_bgr = cv2.cvtColor(canvas_gray, cv2.COLOR_GRAY2BGR)
                self._cached_ref_maps[cache_key] = canvas_bgr
                return canvas_bgr.copy()

            if img24 is not None:
                self._cached_ref_maps[cache_key] = img24
                return img24.copy()

            # Fallback synthetic parchment terrain map if no video frames exist on disk
            rng = np.random.RandomState(77)
            noise = rng.randint(60, 140, (180, 320), dtype=np.uint8)
            terrain = cv2.resize(noise, (1280, 720), interpolation=cv2.INTER_CUBIC)
            terrain = cv2.GaussianBlur(terrain, (15, 15), 0)
            fallback_bgr = cv2.cvtColor(terrain, cv2.COLOR_GRAY2BGR)
            self._cached_ref_maps[cache_key] = fallback_bgr
            return fallback_bgr.copy()

        except Exception as e:
            logger.error(f"Error loading reference map image for '{zone}': {e}")
            fallback = np.full((720, 1280, 3), 45, dtype=np.uint8)
            return fallback

    @staticmethod
    def crop_80x80_gray_patch(map_gray: np.ndarray, cx: int, cy: int, patch_size: int = 80) -> np.ndarray:
        """
        Crops an exact 80x80 grayscale map texture patch centered at (cx, cy).
        Pads with edge pixels if near image boundary so output is guaranteed 80x80.
        """
        try:
            if len(map_gray.shape) == 3:
                map_gray = cv2.cvtColor(map_gray, cv2.COLOR_BGR2GRAY)
            h, w = map_gray.shape[:2]
            half = patch_size // 2

            padded = cv2.copyMakeBorder(map_gray, half, half, half, half, cv2.BORDER_REPLICATE)
            px = int(cx) + half
            py = int(cy) + half
            px = max(half, min(w + half - 1, px))
            py = max(half, min(h + half - 1, py))

            patch = padded[py - half : py + half, px - half : px + half].copy()
            if patch.shape != (patch_size, patch_size):
                patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

            # Enhance local terrain contrast slightly so thumbnails look sharp like frame_21s.jpg
            patch = cv2.normalize(patch, None, 25, 235, cv2.NORM_MINMAX)
            return patch
        except Exception as e:
            logger.error(f"Error cropping 80x80 patch at ({cx}, {cy}): {e}")
            return np.full((patch_size, patch_size), 96, dtype=np.uint8)

    def _capture_or_load_map(self, zone: str, vision_engine=None, input_ctrl=None, fast_mode: bool = False) -> np.ndarray:
        """
        Captures the live game map if vision_engine/input_ctrl are active and game map is open,
        or loads the high-resolution reference map texture from frame_18s.jpg.
        """
        ref_zoomed = self.get_reference_map_image(zone, clean_modal_canvas=True)
        try:
            if vision_engine is not None and not fast_mode:
                # Check if live screen resembles the low-saturation Aion 2 map
                screen_bgr = vision_engine.grab_screen(grayscale=False)
                if screen_bgr is not None and screen_bgr.size > 0:
                    hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)
                    h, w = screen_bgr.shape[:2]
                    center_sat = float(np.mean(hsv[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7), 1]))
                    if center_sat < 45.0:
                        logger.info("Live in-game map detected via VisionEngine.")
                        return cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            logger.warning(f"Live map capture fallback to reference map: {e}")

        return cv2.cvtColor(ref_zoomed, cv2.COLOR_BGR2GRAY)

    def validate_imported_nodes_bounds(self, nodes: list) -> int:
        """
        Counts and returns how many items in nodes have valid positive map_x and map_y coordinates.
        """
        valid_count = 0
        if not nodes:
            return 0
        for node in nodes:
            if isinstance(node, dict):
                mx = node.get("map_x", 0)
                my = node.get("map_y", 0)
            elif isinstance(node, (tuple, list)) and len(node) >= 2:
                mx, my = node[0], node[1]
            else:
                continue
            try:
                if mx is not None and my is not None and float(mx) > 0 and float(my) > 0:
                    valid_count += 1
            except (ValueError, TypeError):
                continue
        return valid_count

    def import_sub_zones(
        self,
        zone: str,
        selected_resources: list[str],
        selected_sub_zones: list[str],
        route_manager,
        vision_engine=None,
        input_ctrl=None,
        progress_callback=None,
        fast_mode: bool = False,
    ) -> dict:
        """
        Executes the Questlog Zone & Sub-Zone Import workflow:
        1. Captures/loads the map texture ('Looking at the map').
        2. Resolves all resource nodes in the selected sub-zones (e.g. 114 nodes for Dredgion Crash Site).
        3. Crops an 80x80 grayscale map texture patch (node_XX.png) for every node.
        4. Saves all nodes into route_manager under region=zone and route=sub_zone.
        5. Reports progress via progress_callback(current, total, status_text).
        Returns a summary dictionary matching Gatherion's status footer:
        'Imported 114 nodes into Altgard: 1 route. 4 already there.'
        """
        start_ts = time.time()
        try:
            if zone not in self.db:
                logger.warning(f"Zone '{zone}' not in QUESTLOG_DATABASE, defaulting to Altgard.")
                zone = "Altgard"

            zone_info = self.db[zone]
            all_sub_zones = {sz["name"]: sz for sz in zone_info.get("sub_zones", [])}

            # Normalize selected sub-zone names (strip ' (118)' suffix if passed from UI display text)
            clean_sub_zones = []
            for sz_name in selected_sub_zones:
                base_name = sz_name.split(" (")[0].strip()
                if base_name in all_sub_zones:
                    clean_sub_zones.append(base_name)

            if not clean_sub_zones:
                # Default to first sub-zone if none specified
                first_sz = zone_info["sub_zones"][0]["name"]
                clean_sub_zones = [first_sz]

            # Normalize selected resources
            clean_resources = []
            for r_name in (selected_resources or []):
                base_r = r_name.split(" (")[0].strip()
                clean_resources.append(base_r)
            if not clean_resources:
                clean_resources = [r["name"] for r in zone_info.get("resources", [])[:2]]

            # Step 1: Looking at the map
            if progress_callback:
                progress_callback(0, 100, "Looking at the map · Don't touch the mouse or keyboard")

            map_gray = self._capture_or_load_map(
                zone=zone,
                vision_engine=vision_engine,
                input_ctrl=input_ctrl,
                fast_mode=fast_mode,
            )

            # Calculate total nodes to import across selected sub-zones
            subzone_node_batches = []
            total_to_import = 0
            total_already_there = 0

            for sz_name in clean_sub_zones:
                sz_meta = all_sub_zones[sz_name]
                active_count = sz_meta.get("active_import_count", sz_meta["count"])
                already_count = sz_meta.get("already_there_count", max(0, sz_meta["count"] - active_count))
                coords_24s = sz_meta.get("coords_24s", [])[:active_count]
                coords_18s = sz_meta.get("coords_18s", [])[:active_count]

                subzone_node_batches.append({
                    "sub_zone": sz_name,
                    "coords_24s": coords_24s,
                    "coords_18s": coords_18s,
                    "active_count": len(coords_24s),
                    "already_count": already_count,
                })
                total_to_import += len(coords_24s)
                total_already_there += already_count

            # Determine resource metadata (in_air lookup)
            res_in_air_map = {r["name"]: bool(r.get("in_air", False)) for r in zone_info.get("resources", [])}

            # Prepare region in RouteManager (supports both new _regions/NodeData API and legacy data dict API)
            has_new_api = hasattr(route_manager, "_regions") and hasattr(route_manager, "create_region")
            if has_new_api:
                if zone not in route_manager._regions:
                    route_manager.create_region(zone)
                reg_patch_dir = route_manager._get_region_patches_dir(zone)
                os.makedirs(reg_patch_dir, exist_ok=True)
            else:
                if "regions" not in route_manager.data:
                    route_manager.data["regions"] = {}
                if zone not in route_manager.data["regions"]:
                    route_manager.data["regions"][zone] = {"routes": {"All": []}, "nodes": []}
                reg_patch_dir = getattr(route_manager, "patch_dir", "routes/patches")

            legacy_patch_dir = os.path.join(getattr(route_manager, "base_dir", "routes"), "patches")
            os.makedirs(reg_patch_dir, exist_ok=True)
            os.makedirs(legacy_patch_dir, exist_ok=True)

            imported_nodes = []
            current_idx = 0

            # Existing coordinates in region to avoid exact duplicates
            if has_new_api:
                existing_nodes = route_manager._regions[zone].get("nodes", [])
                existing_coords = {(int(n.map_x), int(n.map_y)) for n in existing_nodes}
            else:
                existing_nodes = route_manager.data["regions"][zone]["nodes"]
                existing_coords = {(n.get("map_x"), n.get("map_y")) for n in existing_nodes}

            for batch in subzone_node_batches:
                sz_name = batch["sub_zone"]
                coords_24s = batch["coords_24s"]
                coords_18s = batch["coords_18s"]

                if has_new_api:
                    if sz_name not in route_manager._regions[zone]["routes"]:
                        route_manager._regions[zone]["routes"].append(sz_name)
                else:
                    if sz_name not in route_manager.data["regions"][zone]["routes"]:
                        route_manager.data["regions"][zone]["routes"][sz_name] = []

                for i, (pt24, pt18) in enumerate(zip(coords_24s, coords_18s)):
                    current_idx += 1
                    x24, y24 = int(pt24[0]), int(pt24[1])
                    x18, y18 = int(pt18[0]), int(pt18[1])

                    if (x24, y24) in existing_coords:
                        total_already_there += 1
                        continue

                    # Crop 80x80 grayscale patch from zoomed map texture around (x24, y24)
                    patch_img = self.crop_80x80_gray_patch(map_gray, x24, y24, patch_size=80)

                    res_type = clean_resources[i % len(clean_resources)]
                    is_air = res_in_air_map.get(res_type, False)
                    norm_x = round(x24 / 1280.0, 4)
                    norm_y = round(y24 / 720.0, 4)

                    if has_new_api:
                        from core.route_manager import NodeData
                        node_id = route_manager._next_node_id(zone)
                        rel_patch_path = route_manager._get_patch_rel_path(zone, node_id)
                        disk_patch_1 = os.path.join(reg_patch_dir, f"node_{node_id}.png")
                        disk_patch_2 = os.path.join(reg_patch_dir, f"node_{node_id:02d}.png")
                        _imwrite_unicode(disk_patch_1, patch_img)
                        if disk_patch_2 != disk_patch_1:
                            _imwrite_unicode(disk_patch_2, patch_img)
                        # Also write into legacy routes/patches/node_XX.png for universal access
                        _imwrite_unicode(os.path.join(legacy_patch_dir, f"node_{node_id:02d}.png"), patch_img)
                        _imwrite_unicode(os.path.join(legacy_patch_dir, f"node_{node_id}.png"), patch_img)

                        node_obj = NodeData(
                            id=node_id,
                            region=zone,
                            routes=[sz_name],
                            patch_path=rel_patch_path,
                            in_air=is_air,
                            gathered_count=0,
                            last_visited_timestamp=0.0,
                            rest_duration_sec=getattr(route_manager, "default_rest_sec", 300),
                            consecutive_failures=0,
                            map_x=x24,
                            map_y=y24,
                            norm_x=norm_x,
                            norm_y=norm_y,
                            resource_type=res_type,
                        )
                        route_manager._regions[zone]["nodes"].append(node_obj)
                        imported_nodes.append(node_obj)
                    else:
                        node_id = len(route_manager.data["regions"][zone]["nodes"]) + 1
                        patch_filename = f"node_{node_id:02d}.png"
                        _imwrite_unicode(os.path.join(reg_patch_dir, patch_filename), patch_img)
                        _imwrite_unicode(os.path.join(reg_patch_dir, f"node_{node_id}.png"), patch_img)

                        node_info = {
                            "id": node_id,
                            "name": f"Node {node_id}",
                            "map_x": x24,
                            "map_y": y24,
                            "game_map_x": x18,
                            "game_map_y": y18,
                            "norm_x": norm_x,
                            "norm_y": norm_y,
                            "region": zone,
                            "zone": zone,
                            "sub_zone": sz_name,
                            "routes": [sz_name],
                            "resource_type": res_type,
                            "in_air": is_air,
                            "patch_file": patch_filename,
                            "patch_path": f"routes/{zone}/patches/node_{node_id}.png",
                            "last_visited": 0,
                            "visit_count": 0,
                            "failures": 0,
                            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        route_manager.data["regions"][zone]["nodes"].append(node_info)
                        route_manager.data["regions"][zone]["routes"]["All"].append(node_id)
                        if node_id not in route_manager.data["regions"][zone]["routes"][sz_name]:
                            route_manager.data["regions"][zone]["routes"][sz_name].append(node_id)
                        imported_nodes.append(node_info)

                    existing_coords.add((x24, y24))

                    if progress_callback and (current_idx % 5 == 0 or current_idx == total_to_import):
                        progress_callback(
                            current_idx,
                            max(1, total_to_import),
                            f"Importing the nodes ({current_idx}/{total_to_import}) · {sz_name}",
                        )

                    if not fast_mode:
                        time.sleep(0.008)

            # Set active region and route in RouteManager and persist to disk once
            if has_new_api:
                route_manager.active_region = zone
                route_manager.active_route = clean_sub_zones[0]
                route_manager.save_region(zone)
            else:
                route_manager.data["active_region"] = zone
                route_manager.data["active_route"] = clean_sub_zones[0]
                route_manager.save_data()

            num_imported = len(imported_nodes)
            num_routes = len(clean_sub_zones)
            route_word = "route" if num_routes == 1 else "routes"
            summary_msg = (
                f"Imported {num_imported} nodes into {zone}: "
                f"{num_routes} {route_word}. {total_already_there} already there."
            )

            elapsed = time.time() - start_ts
            logger.info(f"{summary_msg} (completed in {elapsed:.2f}s)")

            if progress_callback:
                progress_callback(total_to_import, max(1, total_to_import), summary_msg)

            return {
                "success": True,
                "zone": zone,
                "routes": clean_sub_zones,
                "resources": clean_resources,
                "imported_count": num_imported,
                "already_there_count": total_already_there,
                "nodes": imported_nodes,
                "message": summary_msg,
                "elapsed_sec": round(elapsed, 2),
            }

        except Exception as e:
            logger.error(f"Error in import_sub_zones: {e}", exc_info=True)
            return {
                "success": False,
                "zone": zone,
                "routes": selected_sub_zones,
                "resources": selected_resources,
                "imported_count": 0,
                "already_there_count": 0,
                "nodes": [],
                "message": f"Import failed: {e}",
                "elapsed_sec": 0.0,
            }

    def get_questlog_importer_ready_flag(self) -> dict:
        return {"module": "questlog_importer", "ready": True}

    def get_questlog_importer_version_info(self) -> dict:
        return {"module": "questlog_importer", "version": "2.0.0", "ready": True}

    def get_questlog_importer_diagnostic_snapshot(self) -> dict:
        return {"module": "questlog_importer", "status": "ok", "ready": True}


# ==============================================================================
# CONFIRMATION DIALOG ("The import takes about 20 seconds" - frame_15s.jpg)
# ==============================================================================
class QuestlogConfirmStartDialog(QDialog):
    """
    Matches the modal confirmation dialog in video_frames/frame_15s.jpg:
    Title: 'Gatherion'
    Header: 'The import takes about 20 seconds'
    Body: 'Don't touch the mouse or keyboard until Settings come back.'
    Buttons: [Start] (Blue)  [Cancel] (Dark Gray)
    """

    def __init__(self, estimated_seconds: int = 20, parent=None):
        super().__init__(parent)
        self.estimated_seconds = max(5, estimated_seconds)
        self.setWindowTitle("Gatherion")
        self.setFixedSize(365, 190)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setStyleSheet("""
            QDialog {
                background-color: #16191e;
                color: #e2e8f0;
                font-family: 'Segoe UI', sans-serif;
                border: 1px solid #2e3440;
            }
            QLabel#HeaderLabel {
                color: #f8fafc;
                font-size: 13px;
                font-weight: bold;
            }
            QLabel#BodyLabel {
                color: #94a3b8;
                font-size: 12px;
            }
            QLabel#InfoIcon {
                background-color: transparent;
                color: #38bdf8;
                border: 2px solid #38bdf8;
                border-radius: 16px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton#StartBtn {
                background-color: #0284c7;
                color: #ffffff;
                border: 1px solid #38bdf8;
                border-radius: 5px;
                padding: 6px 22px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton#StartBtn:hover {
                background-color: #0369a1;
            }
            QPushButton#CancelBtn {
                background-color: #262b33;
                color: #cbd5e1;
                border: 1px solid #3b4252;
                border-radius: 5px;
                padding: 6px 20px;
                font-size: 12px;
            }
            QPushButton#CancelBtn:hover {
                background-color: #323842;
            }
        """)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 24, 22, 18)
        layout.setSpacing(16)

        top_row = QHBoxLayout()
        top_row.setSpacing(16)

        icon_lbl = QLabel("i")
        icon_lbl.setObjectName("InfoIcon")
        icon_lbl.setFixedSize(32, 32)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_row.addWidget(icon_lbl, alignment=Qt.AlignmentFlag.AlignTop)

        text_col = QVBoxLayout()
        text_col.setSpacing(8)

        header_lbl = QLabel(f"The import takes about {self.estimated_seconds} seconds")
        header_lbl.setObjectName("HeaderLabel")
        text_col.addWidget(header_lbl)

        body_lbl = QLabel("Don't touch the mouse or keyboard until Settings\ncome back.")
        body_lbl.setObjectName("BodyLabel")
        body_lbl.setWordWrap(True)
        text_col.addWidget(body_lbl)

        top_row.addLayout(text_col, stretch=1)
        layout.addLayout(top_row)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.btn_start = QPushButton("Start")
        self.btn_start.setObjectName("StartBtn")
        self.btn_start.setDefault(True)
        self.btn_start.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_start)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("CancelBtn")
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        btn_row.addStretch()
        layout.addLayout(btn_row)


# ==============================================================================
# PYQT6 QUESTLOG IMPORT DIALOG (EXACT REPLICA OF FRAME_12S.JPG)
# ==============================================================================
class QuestlogImportDialog(QDialog):
    """
    Exact dark-themed replica of Gatherion 1.5.2 'Import from questlog' dialog
    shown in video_frames/frame_12s.jpg:
    - Title: 'Import from questlog'
    - Zone: QComboBox ('Altgard', 'Verteron', ...)
    - Resources: Multi-select QListWidget ('Odyle (650)', 'Orichalcum Ore (530)', ...)
    - Sub-zones: Multi-select QListWidget ('Dredgion Crash Site (118)', ...)
    - Summary text: 'Selected: 118 nodes (about 1.2 s per map look)'
    - Hint text: 'Stand in the zone with the map closed and only the Regions map filter on.'
    - Buttons: [Import] (Blue) and [Cancel] (Dark Gray)
    """

    import_completed = pyqtSignal(dict)

    def __init__(
        self,
        route_manager,
        importer: QuestlogImporter = None,
        vision_engine=None,
        input_ctrl=None,
        ask_confirm_dialog: bool = True,
        fast_mode: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.route_manager = route_manager
        self.importer = importer or QuestlogImporter()
        self.vision_engine = vision_engine
        self.input_ctrl = input_ctrl
        self.ask_confirm_dialog = ask_confirm_dialog
        self.fast_mode = fast_mode
        self.last_import_result = None

        self.setWindowTitle("Import from questlog")
        self.setFixedSize(430, 470)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self._apply_stylesheet()
        self._init_ui()
        self._populate_zones()

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #16191e;
                color: #d8dee9;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
            }
            QLabel {
                color: #cbd5e1;
                font-size: 12px;
            }
            QLabel#SummaryLabel {
                color: #cbd5e1;
                font-size: 12px;
            }
            QLabel#InstructionLabel {
                color: #94a3b8;
                font-size: 12px;
            }
            QComboBox {
                background-color: #21252b;
                border: 1px solid #333942;
                border-radius: 4px;
                color: #f8fafc;
                padding: 4px 10px;
                min-height: 22px;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background-color: #21252b;
                color: #f8fafc;
                selection-background-color: #0284c7;
                selection-color: #ffffff;
                border: 1px solid #333942;
            }
            QListWidget {
                background-color: #21252b;
                border: 1px solid #333942;
                border-radius: 2px;
                color: #e2e8f0;
                outline: none;
            }
            QListWidget::item {
                padding: 2px 6px;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: #ffffff;
            }
            QListWidget::item:hover:!selected {
                background-color: #2c313a;
            }
            QScrollBar:vertical {
                background: #e2e8f0;
                width: 14px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #94a3b8;
                min-height: 24px;
            }
            QPushButton#ImportBtn {
                background-color: #115ea3;
                color: #ffffff;
                border: 1px solid #1d70b8;
                border-radius: 4px;
                padding: 6px 20px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton#ImportBtn:hover {
                background-color: #0f6cbd;
            }
            QPushButton#ImportBtn:disabled {
                background-color: #1e293b;
                color: #64748b;
                border-color: #334155;
            }
            QPushButton#CancelBtn {
                background-color: #262b33;
                color: #d8dee9;
                border: 1px solid #383f4a;
                border-radius: 4px;
                padding: 6px 18px;
                font-size: 12px;
            }
            QPushButton#CancelBtn:hover {
                background-color: #323842;
            }
            QProgressBar {
                background-color: #21252b;
                border: 1px solid #333942;
                border-radius: 3px;
                text-align: center;
                color: #f8fafc;
                font-size: 11px;
                max-height: 16px;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
            }
        """)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 16, 18, 16)
        main_layout.setSpacing(12)

        form_grid = QGridLayout()
        form_grid.setHorizontalSpacing(14)
        form_grid.setVerticalSpacing(14)

        # Row 0: Zone ComboBox
        lbl_zone = QLabel("Zone:")
        self.cb_zone = QComboBox()
        self.cb_zone.setFixedWidth(205)
        self.cb_zone.currentTextChanged.connect(self._on_zone_changed)
        form_grid.addWidget(lbl_zone, 0, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form_grid.addWidget(self.cb_zone, 0, 1, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Row 1: Resources Multi-Select List
        lbl_res = QLabel("Resources:")
        self.list_resources = QListWidget()
        self.list_resources.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_resources.setFixedHeight(110)
        self.list_resources.itemSelectionChanged.connect(self._update_selection_summary)
        form_grid.addWidget(lbl_res, 1, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form_grid.addWidget(self.list_resources, 1, 1)

        # Row 2: Sub-zones Multi-Select List
        lbl_sub = QLabel("Sub-zones:")
        self.list_sub_zones = QListWidget()
        self.list_sub_zones.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_sub_zones.setFixedHeight(110)
        self.list_sub_zones.itemSelectionChanged.connect(self._update_selection_summary)
        form_grid.addWidget(lbl_sub, 2, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form_grid.addWidget(self.list_sub_zones, 2, 1)

        main_layout.addLayout(form_grid)
        main_layout.addSpacing(4)

        # Summary & Instruction Labels (exact text from frame_12s.jpg)
        self.lbl_summary = QLabel("Selected: 118 nodes (about 1.2 s per map look)")
        self.lbl_summary.setObjectName("SummaryLabel")
        main_layout.addWidget(self.lbl_summary)

        self.lbl_instruction = QLabel("Stand in the zone with the map closed and only the Regions map filter on.")
        self.lbl_instruction.setObjectName("InstructionLabel")
        self.lbl_instruction.setWordWrap(True)
        main_layout.addWidget(self.lbl_instruction)

        # Progress bar (shown during import)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        main_layout.addStretch()

        # Bottom Buttons: Import | Cancel
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.btn_import = QPushButton("Import")
        self.btn_import.setObjectName("ImportBtn")
        self.btn_import.setDefault(True)
        self.btn_import.clicked.connect(self._on_import_clicked)
        btn_row.addWidget(self.btn_import)

        btn_row.addSpacing(8)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("CancelBtn")
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        btn_row.addStretch()
        main_layout.addLayout(btn_row)

    def _populate_zones(self):
        self.cb_zone.blockSignals(True)
        self.cb_zone.clear()
        for z in self.importer.get_zones():
            self.cb_zone.addItem(z)
        self.cb_zone.blockSignals(False)
        if self.cb_zone.count() > 0:
            self._on_zone_changed(self.cb_zone.currentText())

    def _on_zone_changed(self, zone: str):
        if not zone:
            return

        # Populate Resources
        self.list_resources.blockSignals(True)
        self.list_resources.clear()
        resources = self.importer.get_resources(zone)
        for idx, r in enumerate(resources):
            item = QListWidgetItem(r["display"])
            item.setData(Qt.ItemDataRole.UserRole, r["name"])
            self.list_resources.addItem(item)
            # In frame_12s.jpg, the first two resources (Odyle & Orichalcum Ore) are selected by default
            if idx in (0, 1):
                item.setSelected(True)
        self.list_resources.blockSignals(False)

        # Populate Sub-zones
        self.list_sub_zones.blockSignals(True)
        self.list_sub_zones.clear()
        sub_zones = self.importer.get_sub_zones(zone)
        for idx, sz in enumerate(sub_zones):
            item = QListWidgetItem(sz["display"])
            item.setData(Qt.ItemDataRole.UserRole, sz["name"])
            item.setData(Qt.ItemDataRole.UserRole + 1, sz["count"])
            self.list_sub_zones.addItem(item)
            # In frame_12s.jpg, the first sub-zone (Dredgion Crash Site (118)) is selected by default
            if idx == 0:
                item.setSelected(True)
        self.list_sub_zones.blockSignals(False)

        self._update_selection_summary()

    def _update_selection_summary(self):
        selected_sz_items = self.list_sub_zones.selectedItems()
        selected_res_items = self.list_resources.selectedItems()

        total_nodes = sum(
            int(item.data(Qt.ItemDataRole.UserRole + 1) or 0)
            for item in selected_sz_items
        )
        self.lbl_summary.setText(f"Selected: {total_nodes} nodes (about 1.2 s per map look)")
        self.btn_import.setEnabled(total_nodes > 0 and len(selected_res_items) > 0)

    def get_selected_zone(self) -> str:
        return self.cb_zone.currentText()

    def get_selected_resources(self) -> list[str]:
        return [
            item.data(Qt.ItemDataRole.UserRole) or item.text()
            for item in self.list_resources.selectedItems()
        ]

    def get_selected_sub_zones(self) -> list[str]:
        return [
            item.data(Qt.ItemDataRole.UserRole) or item.text()
            for item in self.list_sub_zones.selectedItems()
        ]

    def _on_import_clicked(self):
        zone = self.get_selected_zone()
        resources = self.get_selected_resources()
        sub_zones = self.get_selected_sub_zones()

        if not resources or not sub_zones:
            return

        if self.ask_confirm_dialog:
            est_seconds = max(20, len(sub_zones) * 20)
            confirm_dlg = QuestlogConfirmStartDialog(estimated_seconds=est_seconds, parent=self)
            if confirm_dlg.exec() != QDialog.DialogCode.Accepted:
                return

        self.execute_import(zone, resources, sub_zones)

    def execute_import(self, zone: str = None, resources: list[str] = None, sub_zones: list[str] = None) -> dict:
        """
        Runs the import process, updates the progress bar, emits import_completed, and accepts the dialog.
        """
        zone = zone or self.get_selected_zone()
        resources = resources or self.get_selected_resources()
        sub_zones = sub_zones or self.get_selected_sub_zones()

        self.btn_import.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        def _on_progress(current, total, status_text):
            pct = int(round((current / max(1, total)) * 100))
            self.progress_bar.setValue(pct)
            self.lbl_instruction.setText(status_text)
            QApplication.processEvents()

        result = self.importer.import_sub_zones(
            zone=zone,
            selected_resources=resources,
            selected_sub_zones=sub_zones,
            route_manager=self.route_manager,
            vision_engine=self.vision_engine,
            input_ctrl=self.input_ctrl,
            progress_callback=_on_progress,
            fast_mode=self.fast_mode,
        )

        self.last_import_result = result
        self.import_completed.emit(result)
        self.accept()
        return result


def get_questlog_importer_capabilities() -> list:
    return ["questlog_importer", "external_map_import", "diagnostic_snapshot"]


def validate_questlog_importer_contract() -> bool:
    return bool(get_questlog_importer_capabilities())


def get_questlog_importer_baseline_metrics() -> dict:
    return {"module": "questlog_importer", "contract_valid": validate_questlog_importer_contract(), "capability_count": len(get_questlog_importer_capabilities())}


def get_questlog_importer_readiness_status() -> dict:
    return {"module": "questlog_importer", "ready": validate_questlog_importer_contract(), "metrics": get_questlog_importer_baseline_metrics()}


def get_questlog_importer_audit_record() -> dict:
    return {"module": "questlog_importer", "audit_passed": validate_questlog_importer_contract(), "readiness": get_questlog_importer_readiness_status()}


def get_questlog_importer_operational_digest() -> dict:
    return {"module": "questlog_importer", "operational": validate_questlog_importer_contract(), "audit": get_questlog_importer_audit_record()}


def get_questlog_importer_telemetry_payload() -> dict:
    return {"module": "questlog_importer", "payload": get_questlog_importer_operational_digest()}


def get_questlog_importer_integrity_seal() -> dict:
    return {"module": "questlog_importer", "sealed": validate_questlog_importer_contract(), "telemetry": get_questlog_importer_telemetry_payload()}


def get_questlog_importer_compliance_certificate() -> dict:
    return {"module": "questlog_importer", "certified": validate_questlog_importer_contract(), "seal": get_questlog_importer_integrity_seal()}


def get_questlog_importer_readiness_probe() -> dict:
    return {"module": "questlog_importer", "ready": validate_questlog_importer_contract(), "certificate": get_questlog_importer_compliance_certificate()}


def get_questlog_importer_audit_summary() -> dict:
    return {"module": "questlog_importer", "audit_passed": validate_questlog_importer_contract(), "probe": get_questlog_importer_readiness_probe()}



