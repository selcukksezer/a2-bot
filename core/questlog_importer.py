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
from typing import Optional
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QListWidget, QListWidgetItem, QPushButton,
    QMessageBox, QProgressBar, QFrame, QAbstractItemView
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QEventLoop
from PyQt6.QtGui import QFont, QColor, QIcon, QPixmap, QPainter, QPen, QBrush
from version import __version__

logger = logging.getLogger("QuestlogImporter")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Full in-game map canvas (frame_18s.jpg) that node coordinates are stored in.
MAP_WIDTH = 1280
MAP_HEIGHT = 720
# Per-node resource type is NOT known from the questlog data: nodes are placed
# by coordinates only, so invented round-robin assignment is replaced by this.
UNKNOWN_RESOURCE_TYPE = "unknown"


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
        buf = _imencode_cached(ext, img)
        if buf is None:
            return False
        buf.tofile(path)
        return True
    except Exception as e:
        logger.debug(f"_imwrite_unicode failed for '{path}': {e}")
        return False


# Single-entry cache for the last encoded image buffer. import_sub_zones
# writes the identical 80x80 patch to 2-4 paths per node; re-encoding the
# same bytes for every path is pure waste (cv2.imencode of one patch costs
# ~0.4 ms, x228 per legacy import). The key carries the full image content
# (extension + shape + dtype + raw bytes), so a stale hit is impossible,
# and only small images (<= 64 KiB) are cached, bounding memory to one
# small buffer. A caller that mutates the array between writes still gets
# correct bytes because the key is the content itself, not the object id.
_IMWRITE_ENCODE_CACHE: dict = {}
_IMWRITE_ENCODE_CACHE_MAX_BYTES = 65536


def _imencode_cached(ext: str, img: np.ndarray) -> np.ndarray | None:
    """Encode img once per distinct content; return the buffer, or None on failure."""
    if img.nbytes > _IMWRITE_ENCODE_CACHE_MAX_BYTES:
        ok, buf = cv2.imencode(ext, img)
        return buf if ok else None
    cache_key = (ext, img.shape, img.dtype.str, img.tobytes())
    buf = _IMWRITE_ENCODE_CACHE.get(cache_key)
    if buf is None:
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return None
        _IMWRITE_ENCODE_CACHE.clear()
        _IMWRITE_ENCODE_CACHE[cache_key] = buf
    return buf


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
                "verified_coords": True,
                "coords_source": "frame_24s",
                "center": (655, 455),
                "coords_24s": DREDGION_CRASH_SITE_24S_COORDS,
                "coords_18s": _convert_24s_to_18s_coords(DREDGION_CRASH_SITE_24S_COORDS),
            },
            {
                "name": "Nameless Cemetery",
                "count": 118,
                "active_import_count": 118,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (455, 360),
                "coords_24s": _generate_subzone_coords(460, 360, 118, rx=165, ry=110, seed=101),
                "coords_18s": _generate_subzone_coords(455, 360, 118, rx=95, ry=65, seed=101),
            },
            {
                "name": "Fafnite Deposit",
                "count": 112,
                "active_import_count": 112,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (575, 290),
                "coords_24s": _generate_subzone_coords(580, 310, 112, rx=160, ry=105, seed=102),
                "coords_18s": _generate_subzone_coords(575, 290, 112, rx=90, ry=60, seed=102),
            },
            {
                "name": "Basfelt Ruins",
                "count": 111,
                "active_import_count": 111,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (440, 495),
                "coords_24s": _generate_subzone_coords(490, 470, 111, rx=155, ry=105, seed=103),
                "coords_18s": _generate_subzone_coords(440, 495, 111, rx=88, ry=58, seed=103),
            },
            {
                "name": "Sanctum Outpost",
                "count": 83,
                "active_import_count": 83,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
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
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (540, 380),
                "coords_24s": _generate_subzone_coords(620, 380, 124, rx=180, ry=115, seed=201),
                "coords_18s": _generate_subzone_coords(540, 380, 124, rx=105, ry=68, seed=201),
            },
            {
                "name": "Dukaki Settlement",
                "count": 116,
                "active_import_count": 116,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (680, 420),
                "coords_24s": _generate_subzone_coords(680, 410, 116, rx=175, ry=110, seed=202),
                "coords_18s": _generate_subzone_coords(680, 420, 116, rx=98, ry=64, seed=202),
            },
            {
                "name": "Cantas Coast",
                "count": 98,
                "active_import_count": 98,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (480, 290),
                "coords_24s": _generate_subzone_coords(550, 320, 98, rx=160, ry=100, seed=203),
                "coords_18s": _generate_subzone_coords(480, 290, 98, rx=90, ry=58, seed=203),
            },
            {
                "name": "Ardus Shrine",
                "count": 89,
                "active_import_count": 89,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
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
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (520, 370),
                "coords_24s": _generate_subzone_coords(610, 375, 132, rx=185, ry=120, seed=301),
                "coords_18s": _generate_subzone_coords(520, 370, 132, rx=110, ry=72, seed=301),
            },
            {
                "name": "Golden Bough Garrison",
                "count": 115,
                "active_import_count": 115,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (650, 340),
                "coords_24s": _generate_subzone_coords(660, 360, 115, rx=175, ry=110, seed=302),
                "coords_18s": _generate_subzone_coords(650, 340, 115, rx=98, ry=65, seed=302),
            },
            {
                "name": "Eiron Desert",
                "count": 128,
                "active_import_count": 128,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (590, 440),
                "coords_24s": _generate_subzone_coords(635, 420, 128, rx=180, ry=115, seed=303),
                "coords_18s": _generate_subzone_coords(590, 440, 128, rx=105, ry=68, seed=303),
            },
            {
                "name": "Kaidan Headquarters",
                "count": 94,
                "active_import_count": 94,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
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
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (510, 350),
                "coords_24s": _generate_subzone_coords(600, 360, 126, rx=180, ry=115, seed=401),
                "coords_18s": _generate_subzone_coords(510, 350, 126, rx=105, ry=68, seed=401),
            },
            {
                "name": "Patamor Ridge",
                "count": 119,
                "active_import_count": 119,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (640, 390),
                "coords_24s": _generate_subzone_coords(645, 390, 119, rx=175, ry=110, seed=402),
                "coords_18s": _generate_subzone_coords(640, 390, 119, rx=98, ry=64, seed=402),
            },
            {
                "name": "Salintus Desert",
                "count": 134,
                "active_import_count": 134,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (580, 450),
                "coords_24s": _generate_subzone_coords(630, 430, 134, rx=185, ry=120, seed=403),
                "coords_18s": _generate_subzone_coords(580, 450, 134, rx=108, ry=70, seed=403),
            },
            {
                "name": "Mist Mane Village",
                "count": 102,
                "active_import_count": 102,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
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
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (560, 360),
                "coords_24s": _generate_subzone_coords(620, 365, 122, rx=180, ry=115, seed=501),
                "coords_18s": _generate_subzone_coords(560, 360, 122, rx=102, ry=66, seed=501),
            },
            {
                "name": "Senea's Campsite",
                "count": 108,
                "active_import_count": 108,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (650, 415),
                "coords_24s": _generate_subzone_coords(660, 410, 108, rx=170, ry=108, seed=502),
                "coords_18s": _generate_subzone_coords(650, 415, 108, rx=95, ry=62, seed=502),
            },
            {
                "name": "Kishar Observation Post",
                "count": 117,
                "active_import_count": 117,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (690, 330),
                "coords_24s": _generate_subzone_coords(690, 345, 117, rx=175, ry=110, seed=503),
                "coords_18s": _generate_subzone_coords(690, 330, 117, rx=98, ry=64, seed=503),
            },
            {
                "name": "Arbolu's Haven",
                "count": 95,
                "active_import_count": 95,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
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
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (550, 370),
                "coords_24s": _generate_subzone_coords(615, 375, 129, rx=182, ry=116, seed=601),
                "coords_18s": _generate_subzone_coords(550, 370, 129, rx=104, ry=67, seed=601),
            },
            {
                "name": "Red Mane Cavern",
                "count": 115,
                "active_import_count": 115,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (660, 410),
                "coords_24s": _generate_subzone_coords(665, 405, 115, rx=174, ry=110, seed=602),
                "coords_18s": _generate_subzone_coords(660, 410, 115, rx=97, ry=63, seed=602),
            },
            {
                "name": "Hoarfrost Shelter",
                "count": 121,
                "active_import_count": 121,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (620, 315),
                "coords_24s": _generate_subzone_coords(645, 335, 121, rx=178, ry=112, seed=603),
                "coords_18s": _generate_subzone_coords(620, 315, 121, rx=100, ry=65, seed=603),
            },
            {
                "name": "Bakarma Barracks",
                "count": 104,
                "active_import_count": 104,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
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
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (575, 380),
                "coords_24s": _generate_subzone_coords(630, 380, 136, rx=186, ry=120, seed=701),
                "coords_18s": _generate_subzone_coords(575, 380, 136, rx=108, ry=70, seed=701),
            },
            {
                "name": "Observatory Village",
                "count": 114,
                "active_import_count": 114,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (665, 350),
                "coords_24s": _generate_subzone_coords(670, 360, 114, rx=174, ry=110, seed=702),
                "coords_18s": _generate_subzone_coords(665, 350, 114, rx=97, ry=63, seed=702),
            },
            {
                "name": "Jamanok Inn",
                "count": 109,
                "active_import_count": 109,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
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
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (565, 375),
                "coords_24s": _generate_subzone_coords(625, 378, 131, rx=184, ry=118, seed=801),
                "coords_18s": _generate_subzone_coords(565, 375, 131, rx=106, ry=69, seed=801),
            },
            {
                "name": "Baltasar Hill Village",
                "count": 120,
                "active_import_count": 120,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (655, 415),
                "coords_24s": _generate_subzone_coords(660, 405, 120, rx=176, ry=112, seed=802),
                "coords_18s": _generate_subzone_coords(655, 415, 120, rx=99, ry=64, seed=802),
            },
            {
                "name": "Bubu Village",
                "count": 106,
                "active_import_count": 106,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
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
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (540, 440),
                "coords_24s": _generate_subzone_coords(605, 425, 110, rx=172, ry=110, seed=901),
                "coords_18s": _generate_subzone_coords(540, 440, 110, rx=96, ry=62, seed=901),
            },
            {
                "name": "Primum Landing",
                "count": 115,
                "active_import_count": 115,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (660, 290),
                "coords_24s": _generate_subzone_coords(665, 325, 115, rx=175, ry=112, seed=902),
                "coords_18s": _generate_subzone_coords(660, 290, 115, rx=98, ry=64, seed=902),
            },
            {
                "name": "Sulfur Tree Nest",
                "count": 125,
                "active_import_count": 125,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
                "center": (590, 365),
                "coords_24s": _generate_subzone_coords(635, 370, 125, rx=180, ry=115, seed=903),
                "coords_18s": _generate_subzone_coords(590, 365, 125, rx=102, ry=66, seed=903),
            },
            {
                "name": "Siel's Eastern Fortress",
                "count": 96,
                "active_import_count": 96,
                "already_there_count": 0,
                "verified_coords": False,
                "coords_source": "synthetic",
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
        # S-1401: real probe, not a literal True — the importer is only
        # "ready" when its database, verified coordinates and patch
        # cropping actually work. A broken database or a bad coordinate
        # conversion must flip this to False.
        try:
            zones = self.get_zones()
            sub_zones = self.get_sub_zones("Altgard") if "Altgard" in zones else []
            verified = [sz for sz in sub_zones if sz.get("verified_coords", False)]
            dredgion = next(
                (sz for sz in verified if sz["name"] == "Dredgion Crash Site"), None
            )
            dredgion_active = int(dredgion.get("active_import_count", 0)) if dredgion else 0
            dredgion_count = int(dredgion.get("count", 0)) if dredgion else 0
            coords_18s = list(dredgion.get("coords_18s", [])) if dredgion else []
            coords_24s = list(dredgion.get("coords_24s", [])) if dredgion else []
            converted = _convert_24s_to_18s_coords(coords_24s)
            patch = self.crop_80x80_gray_patch(
                np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.uint8), 640, 360
            )
            # The stored coord lists carry `count` entries (118 for
            # Dredgion); only the first `active_import_count` (114) are
            # imported — the rest are the already-there nodes sliced off
            # by import_sub_zones. Probe the active slice.
            active_18s = coords_18s[:dredgion_active]
            dredgion_ok = bool(
                dredgion is not None
                and dredgion_active > 0
                and dredgion_count >= dredgion_active
                and len(coords_18s) == dredgion_count
                and len(coords_24s) == dredgion_count
                and converted[:dredgion_active] == active_18s
                and self.validate_imported_nodes_bounds(active_18s) == dredgion_active
            )
            ready = bool(
                zones
                and verified
                and dredgion_ok
                and patch is not None
                and patch.shape == (80, 80)
            )
            return {
                "questlog_importer_ready": ready,
                "zone_count": len(zones),
                "verified_sub_zone_count": len(verified),
                "dredgion_active_nodes": dredgion_active,
            }
        except Exception as e:
            logger.error(f"Error getting questlog importer health summary: {e}")
            return {
                "questlog_importer_ready": False,
                "zone_count": 0,
                "verified_sub_zone_count": 0,
                "dredgion_active_nodes": 0,
            }

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
        Unknown zones return [] instead of silently returning Altgard's data.
        """
        try:
            zone_data = self.db.get(zone)
            if zone_data is None:
                logger.warning(f"Unknown zone '{zone}': no resources returned (no Altgard fallback).")
                return []
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
        Unknown zones return [] instead of silently returning Altgard's data.
        Each item carries 'verified_coords' (True only for sub-zones whose
        coordinates were extracted from real frames; the rest are synthetic).
        """
        try:
            zone_data = self.db.get(zone)
            if zone_data is None:
                logger.warning(f"Unknown zone '{zone}': no sub-zones returned (no Altgard fallback).")
                return []
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
        The zone parameter is honoured: only zones with a real per-zone map
        file on disk are served. There is no per-zone map for any zone other
        than Altgard in this repo (only video_frames/frame_18s.jpg exists),
        so requesting another zone raises ValueError instead of silently
        returning Altgard's map. No synthetic random-noise map is generated:
        callers must handle the missing-map error honestly.
        """
        if zone not in self.db:
            raise ValueError(f"Unknown zone '{zone}': no reference map available.")
        cache_key = (zone, clean_modal_canvas)
        if cache_key in self._cached_ref_maps:
            return self._cached_ref_maps[cache_key].copy()

        # Per-zone map files, when they exist. Altgard's map is frame_18s.jpg;
        # any other zone needs its own '<zone>_map.jpg' / 'frame_18s_<zone>.jpg'.
        zone_map_candidates = [
            os.path.join(self.base_dir, "video_frames", f"{zone}_map.jpg"),
            os.path.join(self.base_dir, "video_frames", f"frame_18s_{zone}.jpg"),
        ]
        frame18_path = None
        for candidate in zone_map_candidates:
            if os.path.isfile(candidate):
                frame18_path = candidate
                break
        if frame18_path is None:
            if zone == "Altgard":
                frame18_path = os.path.join(self.base_dir, "video_frames", "frame_18s.jpg")
            else:
                raise ValueError(
                    f"No reference map image available for zone '{zone}': "
                    "no per-zone map file exists (Altgard's frame_18s.jpg must "
                    "not be reused for other zones)."
                )
        frame24_path = os.path.join(self.base_dir, "video_frames", "frame_24s.jpg")
        map25_path = os.path.join(self.base_dir, "video_analysis", "map_at_25s.jpg")

        try:
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
                # Extract 1280x720 window aligned with frame_24s coordinate space.
                # Vectorized copy (was a 720-iteration Python row loop doing the
                # same slice assignment one row at a time): the rows copied are
                # exactly y24 in [0, 720) with 164 + y24 < rh, and the columns
                # are [583, min(rw, 1863)) — identical output, one slice.
                canvas_gray = np.full((720, 1280), 35, dtype=np.uint8)
                rh, rw = r18.shape
                rows = max(0, min(720, rh - 164))
                span = max(0, min(rw, 583 + 1280) - 583)
                if rows > 0 and span > 0:
                    canvas_gray[:rows, :span] = r18[164:164 + rows, 583:583 + span]
                canvas_bgr = cv2.cvtColor(canvas_gray, cv2.COLOR_GRAY2BGR)
                self._cached_ref_maps[cache_key] = canvas_bgr
                return canvas_bgr.copy()

            if img24 is not None:
                self._cached_ref_maps[cache_key] = img24
                return img24.copy()

            raise ValueError(
                f"No reference map image could be loaded for zone '{zone}': "
                f"'{frame18_path}' is missing or unreadable."
            )

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error loading reference map image for '{zone}': {e}")
            raise ValueError(f"Could not load reference map image for zone '{zone}': {e}") from e

    @staticmethod
    def crop_80x80_gray_patch(map_gray: np.ndarray, cx: int, cy: int, patch_size: int = 80) -> Optional[np.ndarray]:
        """
        Crops an exact 80x80 grayscale map texture patch centered at (cx, cy).
        Pads with edge pixels if near image boundary so output is guaranteed 80x80.
        Returns None on error (S-0902): callers must skip the node instead
        of importing a fabricated constant-gray patch as if it were real
        map texture.
        """
        try:
            if map_gray is None or not isinstance(map_gray, np.ndarray) or map_gray.size == 0:
                raise ValueError("empty or invalid map image for patch crop")
            if len(map_gray.shape) == 3:
                map_gray = cv2.cvtColor(map_gray, cv2.COLOR_BGR2GRAY)
            h, w = map_gray.shape[:2]
            half = patch_size // 2

            # Clamp the centre into the image (same clamping the old code did
            # via px/py on the padded canvas), crop only the in-bounds part,
            # then replicate-pad just the missing border. The old code padded
            # the WHOLE 1280x720 map with copyMakeBorder on every call — 114
            # full-image copies per import for an 80x80 result. Output is
            # identical: edge pixels are replicated exactly as BORDER_REPLICATE
            # did on the padded canvas.
            # NOTE: the old slice was [c-half : c+half], i.e. 2*half rows even
            # for an odd patch_size (then resized up to patch_size); keep that
            # exact geometry so odd sizes behave byte-identically too.
            eff = half * 2
            cx_c = max(0, min(w - 1, int(cx)))
            cy_c = max(0, min(h - 1, int(cy)))
            x0, y0 = cx_c - half, cy_c - half
            x1, y1 = x0 + eff, y0 + eff
            sx0, sy0 = max(0, x0), max(0, y0)
            sx1, sy1 = min(w, x1), min(h, y1)
            patch = map_gray[sy0:sy1, sx0:sx1]
            pad_top, pad_bottom = sy0 - y0, y1 - sy1
            pad_left, pad_right = sx0 - x0, x1 - sx1
            if pad_top or pad_bottom or pad_left or pad_right:
                patch = cv2.copyMakeBorder(
                    patch, pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_REPLICATE,
                )
            patch = patch.copy()
            if patch.shape != (patch_size, patch_size):
                patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

            # NOTE: no contrast normalization here — stretching the patch to
            # 25..235 destroys the real map texture the patch is meant to
            # preserve for later matching. Return the raw crop.
            return patch
        except Exception as e:
            logger.error(f"Error cropping 80x80 patch at ({cx}, {cy}): {e}")
            # S-0902: no fabricated constant-gray patch — a failed crop
            # is reported as None so the caller skips the node instead
            # of importing fake texture as a successful node.
            return None

    @staticmethod
    def _looks_like_game_map(screen_bgr: np.ndarray) -> bool:
        """
        Stricter live-map check than 'low saturation': a real open map is a
        large, low-saturation image with real texture (not a blank/dark
        screen, loading screen or any other low-saturation UI). Requires
        low saturation in the centre AND overall, plus enough grayscale
        variation that the frame is not uniform.
        """
        try:
            if screen_bgr is None or screen_bgr.size == 0:
                return False
            h, w = screen_bgr.shape[:2]
            if h < 360 or w < 640:
                return False
            hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)
            center_sat = float(np.mean(hsv[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7), 1]))
            overall_sat = float(np.mean(hsv[:, :, 1]))
            gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
            texture = float(np.std(gray))
            return center_sat < 45.0 and overall_sat < 60.0 and texture > 12.0
        except Exception:
            return False

    def _capture_or_load_map(self, zone: str, vision_engine=None, input_ctrl=None, fast_mode: bool = False, return_source: bool = False):
        """
        Captures the live game map if vision_engine/input_ctrl are active and game map is open,
        or loads the high-resolution reference map texture from frame_18s.jpg.
        NOTE: fast_mode only controls the per-node sleep in import_sub_zones;
        it no longer disables live capture (previously the dialog's
        fast_mode=True default meant the live path could never run from UI).

        B-07: referans harita artık "canlı" gibi dönmüyor. Varsayılan
        dönüş (return_source=False) geriye uyumlu ndarray'dir; çağıran
        return_source=True verirse (görüntü, is_live) döner ve referans
        kullanıldığında uyarı loglanır — yamalar bayat referans dokudan
        üretildiyse çağıran bunu bilebilir.
        """
        ref_zoomed = self.get_reference_map_image(zone, clean_modal_canvas=True)
        ref_gray = cv2.cvtColor(ref_zoomed, cv2.COLOR_BGR2GRAY)
        if return_source:
            live_result = lambda img: (img, True)
            ref_result = lambda img: (img, False)
        else:
            live_result = lambda img: img
            ref_result = lambda img: img
        try:
            if vision_engine is not None:
                # Check if live screen resembles the low-saturation Aion 2 map
                screen_bgr = vision_engine.grab_screen(grayscale=False)
                if self._looks_like_game_map(screen_bgr):
                    logger.info("Live in-game map detected via VisionEngine.")
                    return live_result(cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY))
                logger.warning(
                    "Canlı harita yakalanamadı (ekran haritaya benzemiyor); "
                    "referans harita kullanılıyor — yamalar referans dokudan üretilecek."
                )
            else:
                logger.info(
                    "Canlı harita motoru verilmedi; referans harita kullanılıyor "
                    "(yamalar referans dokudan üretilecek)."
                )
        except Exception as e:
            logger.warning(f"Live map capture fallback to reference map: {e}")

        return ref_result(ref_gray)

    def validate_imported_nodes_bounds(self, nodes: list) -> int:
        """
        Counts and returns how many items in nodes have valid map coordinates
        inside the full map canvas (0 <= map_x <= 1280, 0 <= map_y <= 720).
        Merely being > 0 is not enough: coordinates beyond the canvas (or
        non-finite / non-numeric ones) are invalid.
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
                mx_f = float(mx)
                my_f = float(my)
                if not (math.isfinite(mx_f) and math.isfinite(my_f)):
                    continue
                if 0 <= mx_f <= MAP_WIDTH and 0 <= my_f <= MAP_HEIGHT:
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
                # No silent fallback to Altgard: an unknown zone is a failed import.
                msg = f"Import failed: unknown zone '{zone}'."
                logger.error(msg)
                return {
                    "success": False,
                    "zone": zone,
                    "routes": selected_sub_zones,
                    "resources": selected_resources,
                    "imported_count": 0,
                    "already_there_count": 0,
                    "skipped_sub_zones": list(selected_sub_zones or []),
                    "failed_patch_writes": 0,
                    "nodes": [],
                    "message": msg,
                    "elapsed_sec": round(time.time() - start_ts, 2),
                }

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

            # F-1710: only sub-zones with verified (real-frame) coordinates are
            # imported. The other 34 sub-zones carry synthetic coordinates
            # generated by _generate_subzone_coords; importing them would
            # present fabricated nodes as real questlog data, so they are
            # skipped and reported instead of imported.
            skipped_sub_zones = []
            verified_sub_zones = []
            for sz_name in clean_sub_zones:
                if all_sub_zones[sz_name].get("verified_coords", False):
                    verified_sub_zones.append(sz_name)
                else:
                    skipped_sub_zones.append(sz_name)
                    logger.warning(
                        "Skipping sub-zone '%s': coordinates are synthetic "
                        "(not extracted from real frames), not imported.",
                        sz_name,
                    )
            clean_sub_zones = verified_sub_zones
            if not clean_sub_zones:
                msg = (
                    f"Import failed: no selected sub-zone has verified "
                    f"coordinates (skipped: {', '.join(skipped_sub_zones)})."
                )
                logger.error(msg)
                return {
                    "success": False,
                    "zone": zone,
                    "routes": list(selected_sub_zones or []),
                    "resources": selected_resources,
                    "imported_count": 0,
                    "already_there_count": 0,
                    "skipped_sub_zones": skipped_sub_zones,
                    "failed_patch_writes": 0,
                    "nodes": [],
                    "message": msg,
                    "elapsed_sec": round(time.time() - start_ts, 2),
                }

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

            # B-07: dönen görüntünün canlı mı referans mı olduğu bilinir;
            # referanssa sonuç bayraklanır (yamalar bayat dokudan üretilir).
            map_gray, map_is_live = self._capture_or_load_map(
                zone=zone,
                vision_engine=vision_engine,
                input_ctrl=input_ctrl,
                fast_mode=fast_mode,
                return_source=True,
            )
            map_source = "live" if map_is_live else "reference"

            # Calculate total nodes to import across selected sub-zones.
            # already_there is derived ONCE from count - active_import_count
            # (nodes excluded from the batch because they are already there);
            # duplicates found against the route manager during the loop are
            # added to the same counter then, never counted twice (F-1724).
            subzone_node_batches = []
            total_to_import = 0
            total_already_there = 0

            for sz_name in clean_sub_zones:
                sz_meta = all_sub_zones[sz_name]
                active_count = sz_meta.get("active_import_count", sz_meta["count"])
                already_count = max(0, sz_meta["count"] - active_count)
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

            # Progress denominator (F-1733): the 4 'already there' nodes of
            # Dredgion are excluded from coords by the [:active_count] slice,
            # so they were never in total_to_import. Duplicates found against
            # the route manager ARE in the attempted set; they are subtracted
            # from the expected total once existing_coords is known below.
            expected_to_import = total_to_import  # refined after dedup scan

            # NOTE: per-node resource type is unknown from questlog data, so
            # no round-robin assignment is made (F-1712); nodes are stored
            # with resource_type "unknown" and in_air False.

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
            failed_patch_writes = 0

            # Existing coordinates in region to avoid exact duplicates.
            # New-API nodes store full-map (x18, y18) coords, but data written
            # before that fix stored zoomed (x24, y24) coords, so a stored
            # coordinate is ambiguous. The two spaces are therefore tracked
            # in SEPARATE sets: a fresh node's x24 must never be compared
            # against another node's x18 (their ranges overlap and that
            # produces false duplicates).
            if has_new_api:
                existing_nodes = route_manager._regions[zone].get("nodes", [])
                _stored_coords = {(int(n.map_x), int(n.map_y)) for n in existing_nodes}
                seen_18 = set(_stored_coords)
                seen_24 = set(_stored_coords)
            else:
                existing_nodes = route_manager.data["regions"][zone]["nodes"]
                # S-1402: legacy nodes now store full-map (x18, y18) coords
                # (see below), but nodes written before that fix stored
                # zoomed (x24, y24) coords, so a stored coordinate is
                # ambiguous — track both spaces separately, as the new-API
                # branch does, so pre-fix nodes are still recognised as
                # already there and fresh x18 nodes are not compared
                # against stored x24 coords (false duplicates).
                _stored_legacy = set()
                for _n in existing_nodes:
                    try:
                        _stored_legacy.add((int(_n.get("map_x")), int(_n.get("map_y"))))
                    except (TypeError, ValueError, AttributeError):
                        continue
                existing_18 = set(_stored_legacy)
                existing_24 = set(_stored_legacy)

            # F-1733: refine the progress denominator — coords already present
            # in the route manager will be skipped, so they are not "to import".
            _dupes_upfront = 0
            for batch in subzone_node_batches:
                for pt24, pt18 in zip(batch["coords_24s"], batch["coords_18s"]):
                    if has_new_api:
                        if (int(pt18[0]), int(pt18[1])) in seen_18 or (int(pt24[0]), int(pt24[1])) in seen_24:
                            _dupes_upfront += 1
                    elif (int(pt18[0]), int(pt18[1])) in existing_18 or (int(pt24[0]), int(pt24[1])) in existing_24:
                        _dupes_upfront += 1
            expected_to_import = max(0, total_to_import - _dupes_upfront)

            # Legacy branch: next id is max(existing ids) + 1, not len+1,
            # so ids never collide with surviving nodes after deletions (F-1730).
            if not has_new_api:
                _legacy_ids = [int(n.get("id", 0)) for n in existing_nodes
                               if isinstance(n, dict) and str(n.get("id", "")).lstrip("-").isdigit()]
                next_legacy_id = (max(_legacy_ids) + 1) if _legacy_ids else 1

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

                for pt24, pt18 in zip(coords_24s, coords_18s):
                    x24, y24 = int(pt24[0]), int(pt24[1])
                    x18, y18 = int(pt18[0]), int(pt18[1])

                    if has_new_api:
                        if (x18, y18) in seen_18 or (x24, y24) in seen_24:
                            total_already_there += 1
                            continue
                    elif (x18, y18) in existing_18 or (x24, y24) in existing_24:
                        total_already_there += 1
                        continue

                    # Crop 80x80 grayscale patch from zoomed map texture around (x24, y24)
                    patch_img = self.crop_80x80_gray_patch(map_gray, x24, y24, patch_size=80)
                    # S-0902: a failed crop returns None — the node is
                    # skipped, never imported with a fabricated patch.
                    # (Same "not imported" handling as a failed patch
                    # write, F-1729; the legacy branch id is not
                    # consumed because this guard runs before it.)
                    if patch_img is None:
                        failed_patch_writes += 1
                        logger.warning(
                            f"Patch crop failed for node at ({x24}, {y24}) in '{sz_name}'; node skipped, not imported."
                        )
                        continue

                    # F-1712: per-node resource type is unknown from questlog
                    # data — no round-robin fabrication. in_air is unknown too.
                    res_type = UNKNOWN_RESOURCE_TYPE
                    is_air = False

                    if has_new_api:
                        from core.route_manager import NodeData
                        node_id = route_manager._next_node_id(zone)
                        rel_patch_path = route_manager._get_patch_rel_path(zone, node_id)
                        disk_patch_1 = os.path.join(reg_patch_dir, f"node_{node_id}.png")
                        disk_patch_2 = os.path.join(reg_patch_dir, f"node_{node_id:02d}.png")
                        # F-1729: a failed patch write means the node is NOT
                        # imported — it must not be counted or stored.
                        _patch_paths = [disk_patch_1]
                        if disk_patch_2 != disk_patch_1:
                            _patch_paths.append(disk_patch_2)
                        _patch_paths.append(os.path.join(legacy_patch_dir, f"node_{node_id:02d}.png"))
                        _patch_paths.append(os.path.join(legacy_patch_dir, f"node_{node_id}.png"))
                        if not all(_imwrite_unicode(p, patch_img) for p in _patch_paths):
                            failed_patch_writes += 1
                            logger.warning(f"Patch write failed for node {node_id}; node skipped, not imported.")
                            continue
                        # F-1711: store full-map (x18, y18) coords and norms —
                        # the zoomed-modal (x24, y24) space is only for patch
                        # cropping, not for the stored node position.
                        norm_x = round(x18 / MAP_WIDTH, 4)
                        norm_y = round(y18 / MAP_HEIGHT, 4)

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
                            map_x=x18,
                            map_y=y18,
                            norm_x=norm_x,
                            norm_y=norm_y,
                            resource_type=res_type,
                        )
                        route_manager._regions[zone]["nodes"].append(node_obj)
                        imported_nodes.append(node_obj)
                    else:
                        node_id = next_legacy_id
                        next_legacy_id += 1
                        patch_filename = f"node_{node_id:02d}.png"
                        if not all(_imwrite_unicode(os.path.join(reg_patch_dir, p), patch_img)
                                   for p in (patch_filename, f"node_{node_id}.png")):
                            failed_patch_writes += 1
                            next_legacy_id -= 1
                            logger.warning(f"Patch write failed for node {node_id}; node skipped, not imported.")
                            continue
                        # S-1402: store full-map (x18, y18) coords and
                        # norms, like the new-API branch (F-1711) — the
                        # zoomed-modal (x24, y24) space is only for patch
                        # cropping, not for the stored node position.
                        # game_map_x/game_map_y keep the same x18/y18
                        # values for backward compatibility.
                        norm_x = round(x18 / MAP_WIDTH, 4)
                        norm_y = round(y18 / MAP_HEIGHT, 4)

                        node_info = {
                            "id": node_id,
                            "name": f"Node {node_id}",
                            "map_x": x18,
                            "map_y": y18,
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
                        # P3-8 (S-136): "All" rotası yoksa KeyError verirdi.
                        route_manager.data["regions"][zone]["routes"].setdefault("All", []).append(node_id)
                        if node_id not in route_manager.data["regions"][zone]["routes"][sz_name]:
                            route_manager.data["regions"][zone]["routes"][sz_name].append(node_id)
                        imported_nodes.append(node_info)

                    if has_new_api:
                        seen_18.add((x18, y18))
                        seen_24.add((x24, y24))
                    else:
                        existing_18.add((x18, y18))
                        existing_24.add((x24, y24))

                    # Progress counts only nodes actually imported (F-1733).
                    imported_so_far = len(imported_nodes)
                    if progress_callback and (imported_so_far % 5 == 0 or imported_so_far == expected_to_import):
                        progress_callback(
                            imported_so_far,
                            max(1, expected_to_import),
                            f"Importing the nodes ({imported_so_far}/{max(1, expected_to_import)}) · {sz_name}",
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
            if skipped_sub_zones:
                summary_msg += f" Skipped unverified sub-zones: {', '.join(skipped_sub_zones)}."
            if failed_patch_writes:
                summary_msg += f" {failed_patch_writes} node(s) skipped (patch write failed)."
            if not map_is_live:
                summary_msg += " (Harita referans dokudan üretildi — canlı değil.)"

            elapsed = time.time() - start_ts
            logger.info(f"{summary_msg} (completed in {elapsed:.2f}s)")

            if progress_callback:
                progress_callback(num_imported, max(1, expected_to_import), summary_msg)

            # Success means something was actually imported — unless there
            # was nothing to import at all (all nodes already there) and no
            # patch write failed. Zero imports caused by patch failures is
            # a failed import, not an accepted one.
            ok = num_imported > 0 or (expected_to_import == 0 and failed_patch_writes == 0)
            return {
                "success": ok,
                "zone": zone,
                "routes": clean_sub_zones,
                "resources": clean_resources,
                "imported_count": num_imported,
                "already_there_count": total_already_there,
                "skipped_sub_zones": skipped_sub_zones,
                "failed_patch_writes": failed_patch_writes,
                "nodes": imported_nodes,
                "message": summary_msg,
                "elapsed_sec": round(elapsed, 2),
                "map_source": map_source,
                "map_live": bool(map_is_live),
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
                "skipped_sub_zones": [],
                "failed_patch_writes": 0,
                "nodes": [],
                "message": f"Import failed: {e}",
                "elapsed_sec": 0.0,
                "map_source": "unknown",
                "map_live": False,
            }

    def _probe_readiness(self) -> dict:
        """Shared real readiness probe (S-0901): delegates to the S-1401
        health summary — database, verified coordinates and patch
        cropping are actually exercised — instead of a literal True."""
        try:
            return self.get_questlog_importer_health_summary()
        except Exception as e:
            logger.error(f"Error probing questlog importer readiness: {e}")
            return {
                "questlog_importer_ready": False,
                "zone_count": 0,
                "verified_sub_zone_count": 0,
                "dredgion_active_nodes": 0,
            }

    def get_questlog_importer_ready_flag(self) -> dict:
        summary = self._probe_readiness()
        return {"module": "questlog_importer", "ready": bool(summary.get("questlog_importer_ready", False))}

    def get_questlog_importer_version_info(self) -> dict:
        summary = self._probe_readiness()
        return {"module": "questlog_importer", "version": __version__, "ready": bool(summary.get("questlog_importer_ready", False))}

    def get_questlog_importer_diagnostic_snapshot(self) -> dict:
        summary = self._probe_readiness()
        ready = bool(summary.get("questlog_importer_ready", False))
        return {
            "module": "questlog_importer",
            "status": "ok" if ready else "degraded",
            "ready": ready,
            "zone_count": int(summary.get("zone_count", 0)),
            "verified_sub_zone_count": int(summary.get("verified_sub_zone_count", 0)),
            "dredgion_active_nodes": int(summary.get("dredgion_active_nodes", 0)),
        }


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
        self.setFixedSize(400, 200)
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
                font-size: 14px;
                font-weight: bold;
            }
            QLabel#BodyLabel {
                color: #b8c2cf;
                font-size: 13px;
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
                padding: 8px 22px;
                font-size: 12px;
                font-weight: bold;
                min-height: 32px;
            }
            QPushButton#StartBtn:hover {
                background-color: #0369a1;
            }
            QPushButton#CancelBtn {
                background-color: #262b33;
                color: #cbd5e1;
                border: 1px solid #3b4252;
                border-radius: 5px;
                padding: 8px 20px;
                font-size: 12px;
                min-height: 32px;
            }
            QPushButton#CancelBtn:hover {
                background-color: #323842;
            }
        """)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(16)

        top_row = QHBoxLayout()
        top_row.setSpacing(16)

        icon_lbl = QLabel("i")
        icon_lbl.setObjectName("InfoIcon")
        icon_lbl.setFixedSize(32, 32)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_row.addWidget(icon_lbl, alignment=Qt.AlignmentFlag.AlignTop)

        text_col = QVBoxLayout()
        text_col.setSpacing(10)

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
class _ImportWorker(QThread):
    """Runs QuestlogImporter.import_sub_zones off the GUI thread (F-1716).

    The heavy import (map load + per-node patch cropping) previously ran
    inside the import button's slot, freezing the GUI. The worker emits
    progress on its own signal and the finished result; the dialog waits on
    a local QEventLoop so execute_import keeps its synchronous dict return
    while the GUI stays responsive.
    """

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict)

    def __init__(self, importer, zone, resources, sub_zones, route_manager,
                 vision_engine=None, input_ctrl=None, fast_mode=True, parent=None):
        super().__init__(parent)
        self.importer = importer
        self.zone = zone
        self.resources = resources
        self.sub_zones = sub_zones
        self.route_manager = route_manager
        self.vision_engine = vision_engine
        self.input_ctrl = input_ctrl
        self.fast_mode = fast_mode
        self.result = None

    def run(self):
        try:
            self.result = self.importer.import_sub_zones(
                zone=self.zone,
                selected_resources=self.resources,
                selected_sub_zones=self.sub_zones,
                route_manager=self.route_manager,
                vision_engine=self.vision_engine,
                input_ctrl=self.input_ctrl,
                progress_callback=lambda c, t, s: self.progress.emit(c, t, s),
                fast_mode=self.fast_mode,
            )
        except Exception as e:
            self.result = {
                "success": False, "zone": self.zone, "routes": self.sub_zones,
                "resources": self.resources, "imported_count": 0,
                "already_there_count": 0, "skipped_sub_zones": [],
                "failed_patch_writes": 0, "nodes": [],
                "message": f"Import failed: {e}", "elapsed_sec": 0.0,
            }
        self.finished.emit(self.result)


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
        self.setFixedSize(440, 500)
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
                color: #b8c2cf;
                font-size: 12px;
            }
            QComboBox {
                background-color: #21252b;
                border: 1px solid #333942;
                border-radius: 4px;
                color: #f8fafc;
                padding: 4px 10px;
                min-height: 28px;
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
                padding: 4px 8px;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: #ffffff;
            }
            QListWidget::item:hover:!selected {
                background-color: #2c313a;
            }
            QScrollBar:vertical {
                background: #21252b;
                width: 14px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #64748b;
                min-height: 24px;
            }
            QPushButton#ImportBtn {
                background-color: #115ea3;
                color: #ffffff;
                border: 1px solid #1d70b8;
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: bold;
                font-size: 12px;
                min-height: 32px;
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
                padding: 8px 18px;
                font-size: 12px;
                min-height: 32px;
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
                max-height: 18px;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
            }
        """)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 18, 20, 18)
        main_layout.setSpacing(12)

        form_grid = QGridLayout()
        form_grid.setHorizontalSpacing(14)
        form_grid.setVerticalSpacing(16)

        # Row 0: Zone ComboBox
        lbl_zone = QLabel("Zone:")
        self.cb_zone = QComboBox()
        self.cb_zone.setFixedWidth(220)
        self.cb_zone.currentTextChanged.connect(self._on_zone_changed)
        form_grid.addWidget(lbl_zone, 0, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form_grid.addWidget(self.cb_zone, 0, 1, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Row 1: Resources Multi-Select List
        lbl_res = QLabel("Resources:")
        self.list_resources = QListWidget()
        self.list_resources.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_resources.setFixedHeight(120)
        self.list_resources.itemSelectionChanged.connect(self._update_selection_summary)
        form_grid.addWidget(lbl_res, 1, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form_grid.addWidget(self.list_resources, 1, 1)

        # Row 2: Sub-zones Multi-Select List
        lbl_sub = QLabel("Sub-zones:")
        self.list_sub_zones = QListWidget()
        self.list_sub_zones.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_sub_zones.setFixedHeight(120)
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
            item.setData(Qt.ItemDataRole.UserRole + 2, bool(sz.get("verified_coords", False)))
            # S-1403: the number actually imported for a verified sub-zone
            # is active_import_count, not count (import_sub_zones slices
            # coords to active_import_count, e.g. Dredgion 118 -> 114).
            item.setData(Qt.ItemDataRole.UserRole + 3,
                         int(sz.get("active_import_count", sz["count"])))
            self.list_sub_zones.addItem(item)
            # In frame_12s.jpg, the first sub-zone (Dredgion Crash Site (118)) is selected by default
            if idx == 0:
                item.setSelected(True)
        self.list_sub_zones.blockSignals(False)

        self._update_selection_summary()

    def _update_selection_summary(self):
        selected_sz_items = self.list_sub_zones.selectedItems()
        selected_res_items = self.list_resources.selectedItems()

        # Only verified sub-zones are actually imported (F-1710); synthetic
        # ones are skipped, so they must not inflate the node count.
        total_nodes = 0
        unverified_nodes = 0
        for item in selected_sz_items:
            count = int(item.data(Qt.ItemDataRole.UserRole + 1) or 0)
            if item.data(Qt.ItemDataRole.UserRole + 2):
                # S-1403: count only the nodes actually imported — the
                # active_import_count, not the raw count. For Dredgion
                # that is 114, matching import_sub_zones and the
                # _on_import_clicked estimate instead of 118.
                active = item.data(Qt.ItemDataRole.UserRole + 3)
                total_nodes += int(active) if active is not None else count
            else:
                unverified_nodes += count
        summary = f"Selected: {total_nodes} nodes (about 1.2 s per map look)"
        if unverified_nodes:
            summary += f" · {unverified_nodes} unverified nodes will be skipped"
        self.lbl_summary.setText(summary)
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
            # F-1735: never fail silently — tell the user what is missing.
            QMessageBox.warning(
                self,
                "Import from questlog",
                "Select at least one resource and one sub-zone before importing.",
            )
            return

        if self.ask_confirm_dialog:
            # F-1735: estimate from the nodes actually to be imported
            # (verified sub-zones only), so the confirm dialog's seconds
            # agree with the work instead of a fixed '20 seconds' per zone.
            total_nodes = 0
            for sz in self.importer.get_sub_zones(zone):
                if sz["name"] in sub_zones and sz.get("verified_coords", False):
                    total_nodes += int(sz.get("active_import_count", sz["count"]))
            est_seconds = max(5, int(math.ceil(total_nodes * 0.05)) + 2)
            confirm_dlg = QuestlogConfirmStartDialog(estimated_seconds=est_seconds, parent=self)
            if confirm_dlg.exec() != QDialog.DialogCode.Accepted:
                return

        self.execute_import(zone, resources, sub_zones)

    def execute_import(self, zone: str = None, resources: list[str] = None, sub_zones: list[str] = None) -> dict:
        """
        Runs the import process in a worker thread, updates the progress bar,
        emits import_completed, and accepts the dialog ONLY on success
        (F-1716: a failed import must not close as 'accepted').
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

        worker = _ImportWorker(
            importer=self.importer,
            zone=zone,
            resources=resources,
            sub_zones=sub_zones,
            route_manager=self.route_manager,
            vision_engine=self.vision_engine,
            input_ctrl=self.input_ctrl,
            fast_mode=self.fast_mode,
            parent=self,
        )
        worker.progress.connect(_on_progress)
        loop = QEventLoop()
        worker.finished.connect(loop.quit)
        worker.start()
        loop.exec()
        worker.wait()

        result = worker.result or {
            "success": False, "zone": zone, "routes": sub_zones,
            "resources": resources, "imported_count": 0,
            "already_there_count": 0, "skipped_sub_zones": [],
            "failed_patch_writes": 0, "nodes": [],
            "message": "Import failed: worker produced no result.",
            "elapsed_sec": 0.0,
        }

        self.last_import_result = result
        self.import_completed.emit(result)
        if result.get("success"):
            self.imported_summary = result.get("message", "")
            self.accept()
        else:
            # Failed import: stay open, re-enable controls, show the error.
            self.progress_bar.setVisible(False)
            self.btn_import.setEnabled(True)
            self.btn_cancel.setEnabled(True)
            self.lbl_instruction.setText(result.get("message", "Import failed."))
            QMessageBox.critical(
                self, "Import from questlog",
                f"Import failed:\n{result.get('message', 'Unknown error')}",
            )
        return result


def get_questlog_importer_capabilities() -> list:
    return ["questlog_importer", "external_map_import", "diagnostic_snapshot"]


def validate_questlog_importer_contract() -> bool:
    # S-1401 (Decision 3): real probe, not bool() on a non-empty literal
    # capability list — that list can never be empty, so it proves
    # nothing. Delegate to the importer's health summary, which checks
    # the real database, verified coordinates and patch cropping; the
    # whole chain below (baseline/readiness/audit/...) now re-wraps
    # that probe instead of the tautology.
    try:
        capabilities = get_questlog_importer_capabilities()
        if not capabilities or "questlog_importer" not in capabilities:
            return False
        summary = QuestlogImporter().get_questlog_importer_health_summary()
        return bool(summary.get("questlog_importer_ready", False))
    except Exception:
        return False


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



