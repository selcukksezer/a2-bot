# -*- coding: utf-8 -*-
"""
PyQt6 Interactive Numbered Map Viewer (Gatherion 1.5.2 Authentic Replica)
Matches video_frames/frame_24s.jpg and video_frames/frame_27s.jpg:
- Window title: "Nodes on the map of <Region> · green in the route, gray out of it, amber resting"
- Grayscale region map background (auto-crops map region from video_frames/frame_18s.jpg
  or composites saved 80x80 node patches into a mosaic map if no region map is stored)
- High-quality anti-aliased QPainter rendering of all registered region nodes:
  * Green circle (#22c55e) + black border + bold outlined number -> In active route & Ready
  * Amber circle (#f59e0b) + black border + bold outlined number -> Resting (on cooldown)
  * Gray circle (#64748b) + black border + bold outlined number -> Out of active route
  * Blue directional arrow/dot (#3b82f6) -> Player current position (if available)
- Interactive Zoom (mouse wheel) and Pan (mouse drag)
- Left-click on node badge: toggles node in/out of active route (emits node_toggled(int))
- Right-click on node badge: opens node info / reset state / delete context menu
"""

import os
import time
import math
import re
import logging
from typing import Optional, Dict, List, Tuple, Any, Callable, Sequence, Union

from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QMenu, QMessageBox, QSizePolicy, QComboBox
)
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPixmap, QImage,
    QMouseEvent, QWheelEvent, QPaintEvent, QPolygonF, QPainterPath,
    QAction, QCursor, QPalette
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QPointF, QRect, QRectF, QSize

from ui.shell_page_frame import wrap_shell_tab_root_layout
from version import __version__

# P3-6: CWD-bağımlı yollar proje köküne göre çözülür (core/project_root.py).
from core.project_root import PROJECT_ROOT, project_path, resolve_project_path

try:
    from core.route_manager import NodeData, SUBREGION_TO_CANONICAL
except ImportError:
    NodeData = None
    SUBREGION_TO_CANONICAL = {}

logger = logging.getLogger("Gatherion.MapViewer")

# -----------------------------------------------------------------------------
# Altgard HD zone map constants & affine fit (matching ui/cartography_radar.py)
# -----------------------------------------------------------------------------
DEFAULT_MAP_REF: Tuple[float, float] = (1280.0, 720.0)

# Mozaik tuval boyutu (P3-7, tek kaynak): _build_patch_mosaic_pixmap bu
# sabitleri okur; 1180/740 sayısı kodun başka yerinde geçmez.
MOSAIC_CANVAS_WIDTH: int = 1180
MOSAIC_CANVAS_HEIGHT: int = 740

# Başlık çubuğundaki QComboBox geometrisi (P3-7, tek kaynak): başlıktaki iki
# combo (materyal filtresi + harita modu) ile yanlarındaki 30px'lik butonlar
# aynı yükseklikte hizalanır; iki combo da QSS'te padding 5px 10px /
# min-width 160px taşır. Eskiden bg modu combo 4px 8px padding taşıyor ve
# hiçbir combo yükseklik almıyordu.
HEADER_COMBO_MIN_HEIGHT: int = 30
HD_ZONE_MAP_REL = os.path.join("assets", "maps", "altgard.png")

# Bölgeye duyarlı HD harita görselleri (Görev 4; ui/cartography_radar.py ile aynı).
# Source: https://aion2data.com/game/maps/verteron.webp
# AION 2 Verteron dünya haritası (Aion2Data). 2303×2400.
# KURAL: Altgard görseli yalnızca bölge Altgard (veya bilinmiyor) ise adaydır;
# başka bölge için asla Altgard haritası gösterilmez.
HD_ZONE_MAPS: Dict[str, str] = {
    "altgard": os.path.join("assets", "maps", "altgard.png"),
    "verteron": os.path.join("assets", "maps", "verteron.png"),
}

# frame_18s.jpg Altgard dünya haritası kırpmasıdır; yalnızca Altgard
# bölgesinde yedek olabilir. Başka bölgede yedek olamaz.
FRAME18_REGION_KEYS: Tuple[str, ...] = ("", "altgard")


def region_map_candidates(region_name: Optional[str]) -> List[str]:
    """Bölge için HD harita görseli adayları (mutlak yol, sırayla).

    Altgard görseli yalnızca bölge Altgard (veya boş/bilinmiyor) ise adaydır;
    başka bölge için kendi görseli yoksa BOŞ liste döner — çağrıcı mozaik
    yedeğe düşer, yanlış bölge haritası göstermez (Görev 4).
    """
    key = str(region_name or "").strip().lower()
    if key == "":
        # Bölge bilinmiyorsa eski varsayılan: Altgard.
        key = "altgard"
    rel = HD_ZONE_MAPS.get(key)
    if rel is None:
        return []
    return [str(PROJECT_ROOT / rel)]


def is_altgard_map_region(region_name: Optional[str]) -> bool:
    """frame_18s.jpg (Altgard kırpması) bu bölge için yedek olabilir mi?"""
    return str(region_name or "").strip().lower() in FRAME18_REGION_KEYS


def hd_zone_map_path(region: Optional[str] = None) -> str:
    """Bölgeye duyarlı HD harita yolu (region verilmezse Altgard; geri uyumlu)."""
    if region:
        candidates = region_map_candidates(region)
        if candidates:
            return candidates[0]
    return str(PROJECT_ROOT / HD_ZONE_MAP_REL)

_HD_A = 1.46687753
_HD_B = -0.0150533993
_HD_C = -0.00368272413
_HD_D = 1.42560995
_HD_CROP_X = 280.0
_HD_CROP_Y = 80.0
_HD_TX = 242.632119
_HD_TY = 476.526405
HD_FIT_OFFSET: Tuple[float, float] = (
    _HD_TX - _HD_CROP_X * _HD_A - _HD_CROP_Y * _HD_B,
    _HD_TY - _HD_CROP_X * _HD_C - _HD_CROP_Y * _HD_D,
)
_HD_DET = _HD_A * _HD_D - _HD_B * _HD_C


def apply_hd_fit(x: float, y: float) -> Tuple[float, float]:
    """1280x720 video karesi pikseli -> HD Altgard pikseli (2258x2400)."""
    ox, oy = HD_FIT_OFFSET
    return (
        _HD_A * float(x) + _HD_B * float(y) + ox,
        _HD_C * float(x) + _HD_D * float(y) + oy,
    )


def inv_hd_fit(hx: float, hy: float) -> Tuple[float, float]:
    """Altgard HD harita pikseli (2258x2400) -> 1280x720 referans video karesi pikseli."""
    ox, oy = HD_FIT_OFFSET
    dx = float(hx) - ox
    dy = float(hy) - oy
    rx = (_HD_D * dx - _HD_B * dy) / _HD_DET
    ry = (-_HD_C * dx + _HD_A * dy) / _HD_DET
    return rx, ry


def resolve_node_xy(
    node: Any,
    ref_w: float = DEFAULT_MAP_REF[0],
    ref_h: float = DEFAULT_MAP_REF[1],
) -> Tuple[float, float]:
    """
    Düğümü harita çerçevesine oturtur.
    map_x/map_y başka bir çözünürlükte yazıldıysa (norm ile anlaşmıyorsa)
    norm_x/norm_y üzerinden referans çerçeveye çeker. Kayıtlı veriyi değiştirmez.
    """
    try:
        if isinstance(node, dict):
            x = float(node.get("map_x", node.get("x", 0.0)) or 0.0)
            y = float(node.get("map_y", node.get("y", 0.0)) or 0.0)
            nx = float(node.get("norm_x", 0.0) or 0.0)
            ny = float(node.get("norm_y", 0.0) or 0.0)
        else:
            x = float(getattr(node, "map_x", getattr(node, "x", 0.0)) or 0.0)
            y = float(getattr(node, "map_y", getattr(node, "y", 0.0)) or 0.0)
            nx = float(getattr(node, "norm_x", 0.0) or 0.0)
            ny = float(getattr(node, "norm_y", 0.0) or 0.0)
    except Exception as exc:
        logger.debug("Coordinate extraction error: %s", exc)
        return 0.0, 0.0

    if x == 0.0 and y == 0.0 and 0.0 < nx <= 1.0 and 0.0 < ny <= 1.0:
        return nx * float(ref_w), ny * float(ref_h)

    if (
        ref_w > 1.0
        and ref_h > 1.0
        and 0.0 < nx <= 1.0
        and 0.0 < ny <= 1.0
        and abs(x) > 1.0
        and abs(y) > 1.0
    ):
        implied_w = x / nx
        implied_h = y / ny
        if abs(implied_w - ref_w) / ref_w > 0.22 or abs(implied_h - ref_h) / ref_h > 0.22:
            return nx * float(ref_w), ny * float(ref_h)
    return x, y


# Module-level caches for the hot material-matching path (D64): building the
# translation table, compiling the regex and rebuilding the synonym map on
# every call dominated the cost when filtering hundreds of nodes per frame.
_TR_TRANSLATION = str.maketrans("çğıöşü", "cgiosu")
_NON_WORD_RE = re.compile(r"[^\w\s]")
_CATEGORY_SYNONYMS = {
    "orichalcum": ["orichalcum", "ore", "maden", "cevher", "ore_orichalcum"],
    "aria": ["aria", "herb_aria", "ot", "bitki"],
    "mela": ["mela", "herb_mela", "meyve"],
    "targena": ["targena", "herb_targena"],
    "gem": ["gem", "crystal", "odyle", "kristal", "tas", "gem_crystal"],
    "crystal": ["gem", "crystal", "odyle", "kristal", "gem_crystal"],
    "tree": ["tree", "log", "agac", "kereste", "odun", "tree_log", "wood"],
    "log": ["tree", "log", "agac", "kereste", "odun", "tree_log", "wood"],
}


def normalize_material_token(text: Any) -> str:
    """Strips punctuation, emojis, and normalizes Turkish/English characters."""
    if not text:
        return ""
    s = str(text).lower()
    s = s.translate(_TR_TRANSLATION)
    s = _NON_WORD_RE.sub(" ", s)
    return " ".join(s.split())


def node_matches_material(node: Any, filter_val: Union[str, Sequence[str], None]) -> bool:
    """
    Checks if a node matches the selected material filter.
    Returns True if:
    - filter_val is None, empty, 'tumu', 'all', or '🎯 tumu'
    - node matches any material in filter_val (if sequence)
    - node's resource_type, material_type, name, category, or class_name matches filter_val
    """
    if filter_val is None:
        return True

    # If sequence of filters
    if isinstance(filter_val, (list, tuple, set)):
        items = [str(v).strip() for v in filter_val if str(v).strip()]
        if not items:
            return True
        return any(node_matches_material(node, item) for item in items)

    raw_filter = str(filter_val).strip()
    norm_filter = normalize_material_token(raw_filter)
    if not norm_filter or norm_filter in ("tumu", "all", "hepsi", "tum", "tum materyaller"):
        return True

    # Extract all node attributes
    node_tokens: List[str] = []
    if isinstance(node, dict):
        for key in ("resource_type", "material_type", "type", "name", "category", "class_name", "raw_name", "label"):
            val = node.get(key)
            if val:
                node_tokens.append(normalize_material_token(val))
    else:
        for attr in ("resource_type", "material_type", "type", "name", "category", "class_name", "raw_name", "label"):
            val = getattr(node, attr, None)
            if val:
                node_tokens.append(normalize_material_token(val))

    combined_node = " ".join(node_tokens)
    if not combined_node:
        return False

    # Direct substring matches
    if norm_filter in combined_node:
        return True

    # Canonical material category synonyms (module-level table, D64).
    filter_words = norm_filter.split()
    for fw in filter_words:
        if fw in _CATEGORY_SYNONYMS:
            if any(syn in combined_node for syn in _CATEGORY_SYNONYMS[fw]):
                return True
        elif fw in combined_node:
            return True

    return False


def sort_nodes_by_proximity(
    nodes: Sequence[Any],
    start_xy: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Sorts nodes by traveling proximity starting from start_xy (or first node)
    using Greedy Nearest-Neighbor followed by 2-Opt path untangling.
    """
    if not nodes:
        return []

    node_list: List[Dict[str, Any]] = []
    for idx, n in enumerate(nodes):
        if isinstance(n, dict):
            d = dict(n)
        elif hasattr(n, "to_dict") and callable(n.to_dict):
            d = dict(n.to_dict())
        else:
            d = {
                "id": getattr(n, "id", idx + 1),
                "map_x": getattr(n, "map_x", getattr(n, "x", 0.0)),
                "map_y": getattr(n, "map_y", getattr(n, "y", 0.0)),
                "norm_x": getattr(n, "norm_x", 0.0),
                "norm_y": getattr(n, "norm_y", 0.0),
                "resource_type": getattr(n, "resource_type", "Orichalcum Ore"),
                "state": getattr(n, "state", "active"),
                "in_route": getattr(n, "in_route", True),
            }
        node_list.append(d)

    if len(node_list) <= 1:
        return list(node_list)

    # D64: coordinate precomputation. resolve_node_xy does dict/getattr
    # lookups, float conversions and frame checks per call; the old code
    # called it O(n^2) times in greedy + O(iterations * n^2) times in 2-opt.
    # Resolve once per node; all distances below are math.hypot over the
    # same floats, so orderings are bit-identical to the old code.
    coords = [
        resolve_node_xy(item, DEFAULT_MAP_REF[0], DEFAULT_MAP_REF[1])
        for item in node_list
    ]

    unvisited = list(range(len(node_list)))
    ordered: List[Dict[str, Any]] = []
    ordered_coords: List[Tuple[float, float]] = []

    # Select starting node
    if start_xy is not None:
        curr_x, curr_y = float(start_xy[0]), float(start_xy[1])
        best_pos = 0
        best_dist = float("inf")
        for pos, i in enumerate(unvisited):
            nx, ny = coords[i]
            d = math.hypot(nx - curr_x, ny - curr_y)
            if d < best_dist:
                best_dist = d
                best_pos = pos
        pick = unvisited.pop(best_pos)
    else:
        pick = unvisited.pop(0)
    ordered.append(node_list[pick])
    ordered_coords.append(coords[pick])

    while unvisited:
        last_x, last_y = ordered_coords[-1]
        best_pos = 0
        best_dist = float("inf")
        for pos, i in enumerate(unvisited):
            nx, ny = coords[i]
            d = math.hypot(nx - last_x, ny - last_y)
            if d < best_dist:
                best_dist = d
                best_pos = pos
        pick = unvisited.pop(best_pos)
        ordered.append(node_list[pick])
        ordered_coords.append(coords[pick])

    # 2-Opt TSP refinement
    if len(ordered) >= 4:
        n_count = len(ordered)
        # D64: hoisted adjacent segment lengths. seg[j] is the distance
        # ordered[j] -> ordered[j+1]; the current cost of an (i, k) swap is
        # seg[i-1] + seg[k] with no recompute, and only the two boundary
        # segments change on a reversal (inner ones reverse in place,
        # distances are symmetric).
        seg = [
            math.hypot(
                ordered_coords[j + 1][0] - ordered_coords[j][0],
                ordered_coords[j + 1][1] - ordered_coords[j][1],
            )
            for j in range(n_count - 1)
        ]
        improved = True
        iterations = 0
        max_iter = 25
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            for i in range(1, n_count - 1):
                for k in range(i + 1, n_count):
                    d_cur = seg[i - 1]
                    if k + 1 < n_count:
                        d_cur += seg[k]
                    kx, ky = ordered_coords[k]
                    im1x, im1y = ordered_coords[i - 1]
                    d_new = math.hypot(kx - im1x, ky - im1y)
                    if k + 1 < n_count:
                        ix, iy = ordered_coords[i]
                        kp1x, kp1y = ordered_coords[k + 1]
                        d_new += math.hypot(kp1x - ix, kp1y - iy)

                    if d_new < d_cur - 1e-4:
                        ordered[i : k + 1] = reversed(ordered[i : k + 1])
                        ordered_coords[i : k + 1] = reversed(
                            ordered_coords[i : k + 1]
                        )
                        seg[i:k] = reversed(seg[i:k])
                        if i > 0:
                            ax, ay = ordered_coords[i - 1]
                            bx, by = ordered_coords[i]
                            seg[i - 1] = math.hypot(bx - ax, by - ay)
                        if k + 1 < n_count:
                            ax, ay = ordered_coords[k]
                            bx, by = ordered_coords[k + 1]
                            seg[k] = math.hypot(bx - ax, by - ay)
                        improved = True
                        break
                if improved:
                    break

    return ordered


class _MapCanvas(QWidget):
    """
    Interactive Zoom & Pan canvas that draws the grayscale map,
    node badges, outlined node numbers, and player marker.
    """
    node_left_clicked = pyqtSignal(int)
    node_right_clicked = pyqtSignal(int, QPoint)
    node_add_requested = pyqtSignal(float, float)
    node_added = pyqtSignal(dict)
    node_moved = pyqtSignal(int, float, float)
    node_deleted = pyqtSignal(int)
    node_delete_requested = pyqtSignal(int)
    on_canvas_add_node_signal = pyqtSignal(int, float, float)

    # Normalized crop rectangle of the central map inside frame_18s.jpg (works for 1280x720, 1920x1080, etc.)
    # Matches the exact map bounds shown in frame_24s.jpg & frame_27s.jpg (Safe Haven to Calderon Canyon)
    FRAME18_NORM_CROP = (0.295, 0.145, 0.480, 0.625)  # (x_ratio, y_ratio, w_ratio, h_ratio)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(720, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._base_pixmap: Optional[QPixmap] = None
        self._coord_offset: Tuple[int, int] = (0, 0)
        self._nodes: List[Dict[str, Any]] = []
        self._route_node_ids: set = set()
        self._route_node_ids_list: List[int] = []
        self._rest_duration_sec: float = 300.0
        self._player_pos: Optional[Tuple[float, float, float]] = None  # (x, y, angle_deg)
        self._hd_fit: bool = False

        # Zoom & Pan state
        self._zoom: float = 1.0
        self._min_zoom: float = 0.05
        self._max_zoom: float = 5.0
        self._pan: QPointF = QPointF(0.0, 0.0)
        self._is_dragging: bool = False
        self._is_dragging_node: bool = False
        self._dragging_node_id: Optional[int] = None
        self._drag_start_pos: QPointF = QPointF(0.0, 0.0)
        self._drag_moved_px: float = 0.0
        self._hovered_node_id: Optional[int] = None
        self._selected_node_id: Optional[int] = None
        self._initial_fit_done: bool = False
        self._crop_norm_rect: Optional[Tuple[float, float, float, float]] = None
        self.on_canvas_add_node: Optional[Any] = None
        self._patch_cache: Dict[int, QPixmap] = {}

        # Selective material gathering & proximity route
        self._material_filter: Optional[str] = None
        self._proximity_ordered_nodes: List[Dict[str, Any]] = []
        # Cache for the proximity ordering: recomputing NN + 2-Opt on every
        # paintEvent is O(n^2) per frame. Invalidated whenever the node set,
        # the filter, the player position or a node coordinate changes.
        self._proximity_cache_valid: bool = False
        self._proximity_cache_start: Optional[Tuple[float, float]] = None

    def set_material_filter(self, material: Union[str, Sequence[str], None]) -> None:
        """
        Sets the active selective gathering material filter (e.g. 'Orichalcum Ore', 'Aria').
        If None, empty, or 'Tümü', clears the filter and returns to default route rendering.
        """
        if material is None:
            self._material_filter = None
        elif isinstance(material, (list, tuple, set)):
            clean_list = [str(m).strip() for m in material if str(m).strip()]
            if not clean_list or any(normalize_material_token(m) in ("tumu", "all", "") for m in clean_list):
                self._material_filter = None
            else:
                self._material_filter = ", ".join(clean_list)
        else:
            s = str(material).strip()
            norm = normalize_material_token(s)
            if not s or norm in ("tumu", "all", "hepsi", "tum", "tum materyaller"):
                self._material_filter = None
            else:
                self._material_filter = s
        self._proximity_cache_valid = False
        self.calculate_proximity_route()
        self.update()

    def get_material_filter(self) -> Optional[str]:
        return self._material_filter

    def calculate_proximity_route(self, start_xy: Optional[Tuple[float, float]] = None) -> List[Dict[str, Any]]:
        """
        Calculates proximity ordering for matching nodes starting from player position
        (or provided start_xy) using Greedy Nearest-Neighbor + 2-Opt TSP optimization.
        The result is cached: repeated calls (e.g. from every paintEvent) reuse the
        cached ordering until nodes, filter or start position change.
        """
        if not self._material_filter:
            self._proximity_ordered_nodes = []
            self._proximity_cache_valid = False
            self._proximity_cache_start = None
            return []
        if start_xy is None:
            start_xy = (self._player_pos[0], self._player_pos[1]) if self._player_pos is not None else None
        if self._proximity_cache_valid and start_xy == self._proximity_cache_start:
            return list(self._proximity_ordered_nodes)
        matching = [
            n for n in self._nodes
            if isinstance(n, dict) and node_matches_material(n, self._material_filter)
        ]
        if not matching:
            self._proximity_ordered_nodes = []
            self._proximity_cache_start = start_xy
            self._proximity_cache_valid = True
            return []
        self._proximity_ordered_nodes = sort_nodes_by_proximity(matching, start_xy=start_xy)
        self._proximity_cache_start = start_xy
        self._proximity_cache_valid = True
        return list(self._proximity_ordered_nodes)

    def get_proximity_ordered_nodes(self) -> List[Dict[str, Any]]:
        return list(self._proximity_ordered_nodes)

    def filter_nodes_by_selected_material(self, material_query: str = "") -> List[Dict[str, Any]]:
        """
        Filters canvas nodes by the selected material query ('Orichalcum Ore', 'Aria', 'Tümü', etc.),
        synchronizes the canvas filter state, and returns the matching node dictionaries.
        """
        self.set_material_filter(material_query)
        if not self._material_filter:
            return [dict(n) for n in self._nodes if isinstance(n, dict)]
        if self._proximity_ordered_nodes:
            return [dict(n) for n in self._proximity_ordered_nodes if isinstance(n, dict)]
        return [
            dict(n)
            for n in self._nodes
            if isinstance(n, dict) and node_matches_material(n, self._material_filter)
        ]

    def resequence_active_route_by_proximity(
        self,
        start_xy: Optional[Tuple[float, float]] = None,
    ) -> List[int]:
        """
        Resequences the active route nodes starting from start_xy (or player position)
        using Nearest-Neighbor + 2-Opt TSP untangling, updates seq_index (1..N),
        and returns the ordered list of node IDs.
        """
        if start_xy is None and self._player_pos is not None:
            start_xy = (float(self._player_pos[0]), float(self._player_pos[1]))

        if self._material_filter:
            candidates = [
                n for n in self._nodes
                if isinstance(n, dict) and node_matches_material(n, self._material_filter)
            ]
        elif self._route_node_ids:
            candidates = [
                n for n in self._nodes
                if isinstance(n, dict)
                and int(n.get("id", 0)) in self._route_node_ids
                and n.get("in_route") is not False
            ]
        else:
            candidates = [
                n for n in self._nodes
                if isinstance(n, dict) and n.get("in_route", True)
            ]

        if not candidates:
            return []

        ordered_dicts = sort_nodes_by_proximity(candidates, start_xy=start_xy)
        ordered_ids: List[int] = []
        rank_by_id: Dict[int, int] = {}
        for rank, item in enumerate(ordered_dicts, start=1):
            nid = int(item.get("id", 0))
            ordered_ids.append(nid)
            rank_by_id[nid] = rank
            item["seq_index"] = rank

        for node in self._nodes:
            if isinstance(node, dict):
                nid = int(node.get("id", 0))
                if nid in rank_by_id:
                    node["seq_index"] = rank_by_id[nid]

        if self._material_filter:
            self._proximity_ordered_nodes = ordered_dicts
        else:
            self._route_node_ids_list = list(ordered_ids)
            self._route_node_ids = set(ordered_ids)
            # Reorder self._nodes so active route nodes appear in sequence order first
            active_set = set(ordered_ids)
            node_lookup = {int(n.get("id", 0)): n for n in self._nodes if isinstance(n, dict)}
            reordered_nodes = [node_lookup[nid] for nid in ordered_ids if nid in node_lookup]
            for n in self._nodes:
                if isinstance(n, dict) and int(n.get("id", 0)) not in active_set:
                    reordered_nodes.append(n)
            self._nodes = reordered_nodes

        if self._material_filter:
            self._proximity_cache_start = start_xy
            self._proximity_cache_valid = True
        else:
            self._proximity_cache_valid = False
        self.update()
        return ordered_ids

    def _is_ready_for_route_summary(self) -> bool:
        """Real readiness probe for the canvas route summary (S-1803,
        Decision 3: bind the ready flag to a real probe, no unconditional
        True). Ready only when the module contract holds AND a map pixmap
        is loaded AND at least one node is present — a pixmap-less / empty
        canvas is not 'ready'."""
        try:
            if not validate_map_viewer_contract():
                return False
            pix = self._base_pixmap
            if pix is None or pix.isNull():
                return False
            return len(self._nodes) > 0
        except Exception:
            return False

    def get_map_viewer_route_summary(self) -> dict:
        """
        Returns a comprehensive health and node statistics summary for the map canvas.
        """
        now = time.time()
        ready_count = 0
        resting_count = 0
        out_of_route_count = 0
        minimap_auto_count = 0

        for node in self._nodes:
            if not isinstance(node, dict):
                continue
            st = self._classify_node(node, now)
            if st == "ready":
                ready_count += 1
            elif st == "resting":
                resting_count += 1
            else:
                out_of_route_count += 1
            if self._is_minimap_auto_node(node):
                minimap_auto_count += 1

        if self._material_filter:
            filtered_count = sum(
                1 for n in self._nodes
                if isinstance(n, dict) and node_matches_material(n, self._material_filter)
            )
        else:
            filtered_count = len(self._nodes)

        return {
            "map_viewer_ready": self._is_ready_for_route_summary(),
            "total_nodes": len(self._nodes),
            "active_route_nodes": ready_count + resting_count,
            "ready_count": ready_count,
            "resting_count": resting_count,
            "out_of_route_count": out_of_route_count,
            "minimap_auto_count": minimap_auto_count,
            "route_node_ids": list(self._route_node_ids_list),
            "material_filter": self._material_filter or "Tümü",
            "filtered_node_count": filtered_count,
            "hd_fit_active": bool(self._hd_fit),
            "zoom": round(float(self._zoom), 4),
            "player_pos": (
                (float(self._player_pos[0]), float(self._player_pos[1]))
                if self._player_pos is not None
                else None
            ),
            "status_colors": {
                "ready": "#22c55e",
                "resting": "#f59e0b",
                "out_of_route": "#64748b",
            },
        }

    def clear_patch_cache(self) -> None:
        self._patch_cache.clear()

    def set_hd_fit(self, enabled: bool) -> None:
        self._hd_fit = bool(enabled)
        self.update()

    def set_map_pixmap(
        self,
        pixmap: QPixmap,
        coord_offset: Tuple[int, int] = (0, 0),
        crop_norm_rect: Optional[Tuple[float, float, float, float]] = None,
        is_hd: Optional[bool] = None,
    ) -> None:
        self._base_pixmap = pixmap
        self._coord_offset = coord_offset
        self._crop_norm_rect = crop_norm_rect
        self._initial_fit_done = False
        if is_hd is not None:
            self._hd_fit = bool(is_hd)
        elif pixmap is not None and not pixmap.isNull():
            pw = pixmap.width()
            ph = pixmap.height()
            # Size alone misclassifies 4K video frames as HD maps: HD zone maps
            # (e.g. Altgard 2258x2400) are near-square/portrait, video frames are
            # 16:9, so require a map-like aspect ratio as well.
            self._hd_fit = (
                pw >= 1800
                and ph >= 1800
                and crop_norm_rect is None
                and 0.7 <= (pw / ph) <= 1.35
            )
        else:
            self._hd_fit = False
        self._fit_to_view()
        self.update()

    def set_base_pixmap(
        self,
        pixmap: QPixmap,
        coord_offset: Tuple[int, int] = (0, 0),
        crop_norm_rect: Optional[Tuple[float, float, float, float]] = None,
        is_hd: Optional[bool] = None,
    ) -> None:
        """Alias for set_map_pixmap to ensure test and API ergonomics."""
        self.set_map_pixmap(
            pixmap=pixmap,
            coord_offset=coord_offset,
            crop_norm_rect=crop_norm_rect,
            is_hd=is_hd,
        )

    def set_nodes_data(
        self,
        nodes: List[Dict[str, Any]],
        route_node_ids: List[int],
        rest_duration_sec: float = 300.0
    ) -> None:
        self._nodes = list(nodes or [])
        self._route_node_ids_list = [int(nid) for nid in (route_node_ids or [])]
        self._route_node_ids = set(self._route_node_ids_list)
        self._rest_duration_sec = float(rest_duration_sec)
        self._proximity_cache_valid = False
        self.clear_patch_cache()
        self.update()

    def set_player_position(
        self,
        x: Optional[float],
        y: Optional[float],
        angle_deg: float = 0.0
    ) -> None:
        if x is None or y is None:
            self._player_pos = None
        else:
            self._player_pos = (float(x), float(y), float(angle_deg))
        self._proximity_cache_valid = False
        self.update()

    def reset_view(self) -> None:
        self._fit_to_view()
        self.update()

    def _fit_to_view(self) -> None:
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return
        cw = max(1, self.width())
        ch = max(1, self.height())
        pw = max(1, self._base_pixmap.width())
        ph = max(1, self._base_pixmap.height())

        scale_x = cw / pw
        scale_y = ch / ph
        # HD aspect-fit (letterbox/pillarbox): guarantees full map visibility without edge sticking or clipping
        self._zoom = min(scale_x, scale_y)
        self._zoom = max(self._min_zoom, min(self._max_zoom, self._zoom))
        self._pan = QPointF(
            (cw - pw * self._zoom) * 0.5,
            (ch - ph * self._zoom) * 0.5
        )

    def resizeEvent(self, event) -> None:
        old_size = event.oldSize()
        if not self._initial_fit_done and self.width() > 100 and self.height() > 100:
            self._fit_to_view()
            self._initial_fit_done = True
        elif old_size.isValid() and old_size.width() > 0 and old_size.height() > 0:
            # Keep map centered proportionally on dynamic window resize
            dw = (self.width() - old_size.width()) * 0.5
            dh = (self.height() - old_size.height()) * 0.5
            self._pan += QPointF(dw, dh)
        super().resizeEvent(event)

    # -------------------------------------------------------------------------
    # Coordinate Mapping
    # -------------------------------------------------------------------------
    def _node_to_map_coords(
        self,
        raw_x: float,
        raw_y: float,
        norm_x: Optional[float] = None,
        norm_y: Optional[float] = None,
        node: Optional[Dict[str, Any]] = None,
    ) -> QPointF:
        """
        Converts node / player coordinates to pixel coordinates on self._base_pixmap.
        Handles Altgard HD affine fit (apply_hd_fit), frame_18s cropped view,
        dynamic resolution scaling, and eliminates edge-sticking clamping.
        """
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return QPointF(float(raw_x), float(raw_y))

        pw = float(self._base_pixmap.width())
        ph = float(self._base_pixmap.height())

        # 1. Altgard HD Map Mode (apply_hd_fit)
        if self._hd_fit:
            # Check if coordinates are ALREADY in HD map pixel space
            nx = norm_x if norm_x is not None else (node.get("norm_x") if node else None)
            ny = norm_y if norm_y is not None else (node.get("norm_y") if node else None)
            rx_val = float(raw_x)
            ry_val = float(raw_y)
            if nx is not None and ny is not None and float(nx) > 0 and float(ny) > 0:
                fnx, fny = float(nx), float(ny)
                if abs(rx_val - fnx * pw) <= 3.0 and abs(ry_val - fny * ph) <= 3.0:
                    return QPointF(rx_val, ry_val)
            if rx_val > 1280.0 or ry_val > 720.0:
                if (rx_val <= 1920.0 and ry_val <= 1080.0) and node is not None:
                    rx, ry = resolve_node_xy(node, DEFAULT_MAP_REF[0], DEFAULT_MAP_REF[1])
                    hx, hy = apply_hd_fit(rx, ry)
                    return QPointF(hx, hy)
                return QPointF(rx_val, ry_val)

            if node is not None:
                rx, ry = resolve_node_xy(node, DEFAULT_MAP_REF[0], DEFAULT_MAP_REF[1])
            else:
                if nx is not None and ny is not None and (0.0 < float(nx) <= 1.0 or 0.0 < float(ny) <= 1.0):
                    rx = float(nx) * DEFAULT_MAP_REF[0]
                    ry = float(ny) * DEFAULT_MAP_REF[1]
                elif 0.0 < rx_val <= 1.0 and 0.0 < ry_val <= 1.0:
                    rx = rx_val * DEFAULT_MAP_REF[0]
                    ry = ry_val * DEFAULT_MAP_REF[1]
                elif rx_val <= DEFAULT_MAP_REF[0] and ry_val <= DEFAULT_MAP_REF[1]:
                    rx = rx_val
                    ry = ry_val
                elif rx_val <= 1920.0 and ry_val <= 1080.0:
                    rx = rx_val * (DEFAULT_MAP_REF[0] / 1920.0)
                    ry = ry_val * (DEFAULT_MAP_REF[1] / 1080.0)
                else:
                    return QPointF(rx_val, ry_val)
            hx, hy = apply_hd_fit(rx, ry)
            return QPointF(hx, hy)

        # 2. Extract norm_x / norm_y if node dict was passed
        if node is not None:
            if norm_x is None:
                nx_val = node.get("norm_x")
                if nx_val is not None:
                    try:
                        norm_x = float(nx_val)
                    except (ValueError, TypeError):
                        pass
            if norm_y is None:
                ny_val = node.get("norm_y")
                if ny_val is not None:
                    try:
                        norm_y = float(ny_val)
                    except (ValueError, TypeError):
                        pass

        # 3. Check if raw_x / raw_y are themselves normalized (0.0 < val <= 1.0)
        if norm_x is None and norm_y is None:
            if 0.0 < float(raw_x) <= 1.0 and 0.0 < float(raw_y) <= 1.0:
                norm_x = float(raw_x)
                norm_y = float(raw_y)

        # 4. Check for crop normalized window (e.g. FRAME18_NORM_CROP = 0.295, 0.145, 0.480, 0.625)
        crop_norm = getattr(self, "_crop_norm_rect", None)
        ox, oy = self._coord_offset

        if crop_norm is not None:
            rx, ry, rw, rh = crop_norm
            if norm_x is None or norm_y is None:
                norm_x = float(raw_x) / DEFAULT_MAP_REF[0]
                norm_y = float(raw_y) / DEFAULT_MAP_REF[1]
            if rw > 0 and rh > 0:
                u = (norm_x - rx) / rw
                v = (norm_y - ry) / rh
                mapped_x = u * pw
                mapped_y = v * ph
                # No border clamping: prevents corner-piling / kenara yapışma
                return QPointF(mapped_x, mapped_y)

        # 5. Full map / mosaic / standard resolution
        if norm_x is not None and norm_y is not None and (0.0 < norm_x <= 1.0 or 0.0 < norm_y <= 1.0):
            return QPointF(norm_x * pw, norm_y * ph)

        rx_val = float(raw_x)
        ry_val = float(raw_y)

        if ox != 0 or oy != 0:
            if rx_val >= ox and ry_val >= oy:
                return QPointF(rx_val - ox, ry_val - oy)

        if rx_val > pw * 1.15 or ry_val > ph * 1.15:
            if rx_val > 1280 or ry_val > 720:
                scale_x = pw / 1920.0
                scale_y = ph / 1080.0
            else:
                scale_x = pw / 1280.0
                scale_y = ph / 720.0
            rx_val *= scale_x
            ry_val *= scale_y

        return QPointF(rx_val, ry_val)

    def _map_to_widget(self, map_pt: QPointF) -> QPointF:
        return QPointF(
            self._pan.x() + map_pt.x() * self._zoom,
            self._pan.y() + map_pt.y() * self._zoom,
        )

    def _widget_to_map(self, widget_pt: QPointF) -> QPointF:
        z = max(0.001, self._zoom)
        return QPointF(
            (widget_pt.x() - self._pan.x()) / z,
            (widget_pt.y() - self._pan.y()) / z,
        )

    def _map_to_node_coords(self, map_pt: QPointF) -> Tuple[int, int, float, float]:
        """
        Converts pixel coordinate on self._base_pixmap back to (map_x, map_y, norm_x, norm_y)
        relative to the standard 1280x720 video frame.
        Uses inv_hd_fit when HD map mode is active without clamping to prevent edge sticking.
        """
        if self._hd_fit:
            rx, ry = inv_hd_fit(map_pt.x(), map_pt.y())
            # norm, 1280x720 referans kareye göredir (pixmap kesrine değil):
            # _node_to_map_coords norm'u DEFAULT_MAP_REF ile çarpar, bu yüzden
            # ters yön de rx/1280, ry/720 olmalı; aksi halde gidiş-dönüş
            # tutmaz (520 -> norm 0.2618 olur, 0.40625 beklenir).
            norm_x = max(0.0, min(1.0, rx / DEFAULT_MAP_REF[0]))
            norm_y = max(0.0, min(1.0, ry / DEFAULT_MAP_REF[1]))
            map_x = int(round(rx))
            map_y = int(round(ry))
            return map_x, map_y, float(norm_x), float(norm_y)

        crop_norm = getattr(self, "_crop_norm_rect", None)
        pw = max(1.0, float(self._base_pixmap.width())) if (self._base_pixmap and not self._base_pixmap.isNull()) else DEFAULT_MAP_REF[0]
        ph = max(1.0, float(self._base_pixmap.height())) if (self._base_pixmap and not self._base_pixmap.isNull()) else DEFAULT_MAP_REF[1]

        if crop_norm is not None:
            rx, ry, rw, rh = crop_norm
            u = map_pt.x() / pw
            v = map_pt.y() / ph
            norm_x = max(0.0, min(1.0, rx + u * rw))
            norm_y = max(0.0, min(1.0, ry + v * rh))
            map_x = int(round(norm_x * DEFAULT_MAP_REF[0]))
            map_y = int(round(norm_y * DEFAULT_MAP_REF[1]))
            return map_x, map_y, float(norm_x), float(norm_y)

        norm_x = max(0.0, min(1.0, map_pt.x() / pw))
        norm_y = max(0.0, min(1.0, map_pt.y() / ph))
        map_x = int(round(norm_x * DEFAULT_MAP_REF[0]))
        map_y = int(round(norm_y * DEFAULT_MAP_REF[1]))
        return map_x, map_y, float(norm_x), float(norm_y)

    def _widget_to_node_coords(self, widget_pt: QPointF) -> Tuple[int, int, float, float]:
        """Converts widget pixel position to (map_x, map_y, norm_x, norm_y) on reference frame."""
        map_pt = self._widget_to_map(widget_pt)
        return self._map_to_node_coords(map_pt)

    def _find_node_at_widget_pos(self, pos: QPointF, hit_radius: float = 13.0) -> Optional[int]:
        """Finds the closest node within hit_radius (in widget pixels)."""
        best_id: Optional[int] = None
        best_dist = hit_radius
        for node in reversed(self._nodes):
            try:
                nid = int(node.get("id", 0))
                nx = float(node.get("map_x", node.get("x", 0)))
                ny = float(node.get("map_y", node.get("y", 0)))
                norm_x = node.get("norm_x")
                norm_y = node.get("norm_y")
                map_pt = self._node_to_map_coords(nx, ny, norm_x=norm_x, norm_y=norm_y, node=node)
                w_pt = self._map_to_widget(map_pt)
                dist = math.hypot(pos.x() - w_pt.x(), pos.y() - w_pt.y())
                if dist <= best_dist:
                    best_dist = dist
                    best_id = nid
            except Exception:
                continue
        return best_id

    def _create_node_at_coords(
        self,
        map_x: float,
        map_y: float,
        norm_x: float,
        norm_y: float,
    ) -> int:
        """Creates a new node at the specified coordinates, saves to RouteManager if available, and updates canvas."""
        parent = self.parent()
        if parent is not None and hasattr(parent, "add_node") and callable(parent.add_node):
            try:
                node_dict = parent.add_node(map_x, map_y, norm_x=norm_x, norm_y=norm_y)
                if isinstance(node_dict, dict) and "id" in node_dict:
                    new_id = int(node_dict["id"])
                    self.node_add_requested.emit(float(map_x), float(map_y))
                    self.on_canvas_add_node_signal.emit(new_id, float(map_x), float(map_y))
                    self.node_added.emit(node_dict)
                    return new_id
            except Exception as exc:
                logger.error("Error creating node via parent.add_node: %s", exc)

        routes = getattr(parent, "routes", None)
        region = getattr(parent, "_region_name", "Altgard") or "Altgard"
        route = getattr(parent, "_route_name", "All") or "All"
        routes_list = [route] if (route and route != "All") else []
        new_id: Optional[int] = None

        if routes is not None and hasattr(routes, "add_node"):
            try:
                node_obj = routes.add_node(
                    region=region,
                    route=route,
                    routes=routes_list,
                    map_x=int(round(map_x)),
                    map_y=int(round(map_y)),
                    norm_x=float(norm_x),
                    norm_y=float(norm_y),
                    source_type="manual",
                )
                if node_obj is not None:
                    new_id = getattr(node_obj, "id", None)
                if hasattr(routes, "save_region"):
                    routes.save_region(region)
                elif hasattr(routes, "save_data"):
                    routes.save_data()
            except Exception as exc:
                logger.error("Error creating node via routes.add_node: %s", exc)

        if new_id is None:
            existing_ids = [int(n.get("id", 0)) for n in self._nodes if isinstance(n, dict)]
            new_id = (max(existing_ids) + 1) if existing_ids else 1

        new_node = {
            "id": new_id,
            "region": region,
            "routes": list(routes_list),
            # Repo-göreli saklanır (kalıcı JSON makineden bağımsız kalsın);
            # okuyan taraf resolve_node_patch_path ile köke göre çözer (P3-6).
            "patch_path": f"routes/{region}/patches/node_{new_id}.png",
            "map_x": int(round(map_x)),
            "map_y": int(round(map_y)),
            "norm_x": float(norm_x),
            "norm_y": float(norm_y),
            "source_type": "manual",
            "state": "active",
            "max_harvests": 3,
            "current_harvests": 0,
            "in_route": True,
            "seq_index": len(self._nodes) + 1,
        }
        if not any(int(n.get("id", -1)) == new_id for n in self._nodes):
            self._nodes.append(new_node)
            self._route_node_ids.add(new_id)
            self._route_node_ids_list.append(new_id)
            self._proximity_cache_valid = False

        if callable(getattr(self, "on_canvas_add_node", None)):
            try:
                self.on_canvas_add_node(new_id, map_x, map_y)
            except Exception as exc:
                logger.debug("on_canvas_add_node callback error: %s", exc)

        self.node_add_requested.emit(float(map_x), float(map_y))
        self.on_canvas_add_node_signal.emit(new_id, float(map_x), float(map_y))
        self.node_added.emit(new_node)
        self.update()
        logger.info("Yeni nokta eklendi: #%d (%d, %d)", new_id, int(round(map_x)), int(round(map_y)))
        return new_id

    def _save_node_position(
        self,
        node_id: int,
        map_x: float,
        map_y: float,
        norm_x: float,
        norm_y: float,
    ) -> bool:
        """Persists the updated node position to RouteManager.

        K-19: ``node_moved`` yalnızca konum gerçekten kalıcı depoya
        yazıldıysa (veya kalıcı depo hiç yoksa — o zaman tuval tek
        depodur) yayılır. Yazım başarısızsa sinyal yayılmaz; dinleyenler
        eski konumu korur.
        """
        self._proximity_cache_valid = False
        # [O4] The canvas is reparented into an inner host by the shell
        # chrome (D32 F-0689: wrap_shell_tab_root_layout returns an inner
        # layout for non-embedded pages), so self.parent() is a plain
        # QWidget, not the MapViewerWindow. Walk up the ancestor chain to
        # the owner that has move_node / routes / _region_name.
        owner: Optional[QWidget] = self.parent()
        while owner is not None and not callable(getattr(owner, "move_node", None)):
            owner = owner.parentWidget() if hasattr(owner, "parentWidget") else None
        if owner is not None:
            try:
                moved = owner.move_node(node_id, map_x, map_y, norm_x=norm_x, norm_y=norm_y)
                # move_node kendisi node_moved yayar ve yazım sonucunu
                # döner; False temiz bir ret ise ikinci bir daldan
                # yeniden yazmak (çift dal) yerine burada durulur.
                # Başarıda tuval sinyali de yayılır (önceki davranış;
                # pencere slotu aynı koordinatı bellekten günceller).
                if moved is None or bool(moved):
                    self.node_moved.emit(node_id, float(map_x), float(map_y))
                    return True
                logger.error(
                    "Taşınan düğüm #%d kalıcı depoya yazılamadı; node_moved yayılmadı.",
                    node_id,
                )
                return False
            except Exception as exc:
                logger.error("Error delegating to parent.move_node: %s", exc)

        routes = None
        region = "Altgard"
        node = self.parent()
        while node is not None:
            if routes is None:
                routes = getattr(node, "routes", None)
            name = getattr(node, "_region_name", None)
            if name:
                region = name
                break
            node = node.parentWidget() if hasattr(node, "parentWidget") else None
        if routes is not None:
            persist_ok = False
            try:
                updated = False
                if hasattr(routes, "_regions") and isinstance(routes._regions, dict) and region in routes._regions:
                    for n in routes._regions[region].get("nodes", []):
                        cur_id = getattr(n, "id", None) or (n.get("id") if isinstance(n, dict) else None)
                        if cur_id is not None and int(cur_id) == node_id:
                            if hasattr(n, "map_x"):
                                n.map_x = int(round(map_x))
                                n.map_y = int(round(map_y))
                                n.norm_x = float(norm_x)
                                n.norm_y = float(norm_y)
                            elif isinstance(n, dict):
                                n["map_x"] = int(round(map_x))
                                n["map_y"] = int(round(map_y))
                                n["norm_x"] = float(norm_x)
                                n["norm_y"] = float(norm_y)
                            updated = True
                            break
                    if updated:
                        if hasattr(routes, "save_region"):
                            persist_ok = routes.save_region(region) is not False
                        elif hasattr(routes, "save_data"):
                            persist_ok = routes.save_data() is not False
                elif hasattr(routes, "data") and isinstance(routes.data, dict):
                    reg_nodes = routes.data.get("regions", {}).get(region, {}).get("nodes", [])
                    for n in reg_nodes:
                        cur_id = n.get("id") if isinstance(n, dict) else getattr(n, "id", None)
                        if cur_id is not None and int(cur_id) == node_id:
                            n["map_x"] = int(round(map_x))
                            n["map_y"] = int(round(map_y))
                            n["norm_x"] = float(norm_x)
                            n["norm_y"] = float(norm_y)
                            updated = True
                            break
                    if updated and hasattr(routes, "save_data"):
                        persist_ok = routes.save_data() is not False
            except Exception as exc:
                logger.error("Failed persisting moved node #%d: %s", node_id, exc)
                persist_ok = False
            if not persist_ok:
                logger.error(
                    "Taşınan düğüm #%d kalıcı depoya yazılamadı (%s); node_moved yayılmadı.",
                    node_id, region,
                )
                return False
        else:
            logger.debug(
                "Taşınan düğüm #%d için kalıcı depo yok; konum yalnızca tuval belleğinde.",
                node_id,
            )

        self.node_moved.emit(node_id, float(map_x), float(map_y))
        return True

    def delete_node(self, node_id: int) -> bool:
        """Deletes a node from the canvas and route manager.

        Returns True only if the node was actually removed; False when the
        node does not exist (or the parent reports failure). The deletion
        runs exactly once: when a parent handler removes the node, this
        method does not re-delete via the node_deleted signal (the window
        slot guards against already-deleted ids).
        """
        parent = self.parent()
        parent_ok: Optional[bool] = None
        if parent is not None and hasattr(parent, "delete_node") and callable(parent.delete_node):
            try:
                res = parent.delete_node(node_id)
                parent_ok = True if res is None else bool(res)
            except Exception as exc:
                logger.error("Error in parent.delete_node: %s", exc)
                parent_ok = False
        elif parent is not None and hasattr(parent, "_delete_node") and callable(parent._delete_node):
            try:
                res = parent._delete_node(node_id)
                parent_ok = True if res is None else bool(res)
            except Exception as exc:
                logger.error("Error in parent._delete_node: %s", exc)
                parent_ok = False
        existed = any(int(n.get("id", -1)) == node_id for n in self._nodes if isinstance(n, dict))
        if parent_ok is False:
            # F-0574: the parent explicitly failed to delete the node, so
            # the canvas must not pretend the deletion happened (no local
            # removal, no node_deleted emit, return False).
            return False
        if parent_ok is None and not existed:
            # F-0574: no parent handled the delete and the node is not on
            # this canvas at all — nothing was removed.
            return False
        # P3-8 (S-040): _nodes dict olmayabilir; .get çağrısı AttributeError
        # verirdi. Dict-olmayan öğe hiçbir id ile eşleşemez, korunur.
        kept_nodes = []
        for n in self._nodes:
            if isinstance(n, dict):
                try:
                    nid = int(n.get("id", -1))
                except (TypeError, ValueError):
                    nid = -1
                if nid == node_id:
                    continue
            kept_nodes.append(n)
        self._nodes = kept_nodes
        self._route_node_ids.discard(node_id)
        if node_id in self._route_node_ids_list:
            self._route_node_ids_list.remove(node_id)
        self._proximity_cache_valid = False
        self.node_deleted.emit(node_id)
        self.update()
        return True

    def _show_default_context_menu(self, node_id: int, global_pos: QPoint) -> None:
        """Fallback context menu if parent does not handle node_right_clicked."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #181b20;
                color: #e5e7eb;
                border: 1px solid #2d323b;
                border-radius: 6px;
                padding: 4px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }
            QMenu::item {
                padding: 8px 22px 8px 14px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #2563eb;
                color: #ffffff;
            }
        """)
        act_toggle = QAction(f"Rotaya Ekle / Çıkar (#{node_id})", menu)
        act_toggle.triggered.connect(lambda: self.node_left_clicked.emit(node_id))
        menu.addAction(act_toggle)
        menu.addSeparator()
        act_del = QAction(f"Noktayı Sil (#{node_id})", menu)
        # S038 (v19/P3-8): menüden silme onaysızdı — yanlışlıkla bir
        # tıklama düğümü kalıcı siliyordu. Onay sorulur (ayarlar
        # penceresindeki toplu-silme deseniyle aynı). Shift+Sağ Tık
        # kısayolu bilinçli "hızlı silme"dir — test kilitli
        # (test_right_click_delete_node), orada onay sorulmaz.
        def _confirm_delete() -> None:
            reply = QMessageBox.question(
                self,
                "Noktayı Sil",
                f"#{node_id} numaralı düğüm silinsin mi?\nBu işlem geri alınamaz.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.delete_node(node_id)

        act_del.triggered.connect(_confirm_delete)
        menu.addAction(act_del)
        menu.popup(global_pos)

    # -------------------------------------------------------------------------
    # Mouse Events (Zoom, Pan, Drag Node, Add Node, Delete Node)
    # -------------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:
        try:
            angle_delta = event.angleDelta().y()
            if angle_delta == 0:
                return
            factor = 1.15 if angle_delta > 0 else (1.0 / 1.15)
            new_zoom = max(self._min_zoom, min(self._max_zoom, self._zoom * factor))
            if abs(new_zoom - self._zoom) < 1e-5:
                return

            cursor_pos = event.position()
            map_pt_before = self._widget_to_map(cursor_pos)
            self._zoom = new_zoom
            self._pan = QPointF(
                cursor_pos.x() - map_pt_before.x() * self._zoom,
                cursor_pos.y() - map_pt_before.y() * self._zoom,
            )
            self.update()
        except Exception as exc:
            logger.error("Error in MapCanvas.wheelEvent: %s", exc)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            hit_id = self._find_node_at_widget_pos(event.position(), hit_radius=13.0)
            if hit_id is not None:
                self._is_dragging_node = True
                self._dragging_node_id = hit_id
                self._is_dragging = False
                self._drag_start_pos = event.position()
                self._drag_moved_px = 0.0
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self._is_dragging_node = False
                self._dragging_node_id = None
                self._is_dragging = True
                self._drag_start_pos = event.position()
                self._drag_moved_px = 0.0
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._is_dragging_node = False
            self._dragging_node_id = None
            self._is_dragging = True
            self._drag_start_pos = event.position()
            self._drag_moved_px = 0.0
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            nid = self._find_node_at_widget_pos(event.position(), hit_radius=13.0)
            if nid is not None:
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self.delete_node(nid)
                else:
                    self.node_right_clicked.emit(nid, event.globalPosition().toPoint())
                    if self.receivers(self.node_right_clicked) <= 0:
                        self._show_default_context_menu(nid, event.globalPosition().toPoint())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._is_dragging_node and self._dragging_node_id is not None:
            delta = pos - self._drag_start_pos
            self._drag_moved_px += math.hypot(delta.x(), delta.y())
            self._drag_start_pos = pos

            map_x, map_y, norm_x, norm_y = self._widget_to_node_coords(pos)
            for node in self._nodes:
                if int(node.get("id", -1)) == self._dragging_node_id:
                    node["map_x"] = int(round(map_x))
                    node["map_y"] = int(round(map_y))
                    node["norm_x"] = float(norm_x)
                    node["norm_y"] = float(norm_y)
                    break
            # Live update only: emit node_moved so the UI follows the drag,
            # but do NOT write to disk per mouse-move — the window slot updates
            # the position in memory and the single disk save happens on
            # release via _save_node_position().
            self.node_moved.emit(self._dragging_node_id, float(map_x), float(map_y))
            self._proximity_cache_valid = False
            self.update()
            return

        if self._is_dragging:
            delta = pos - self._drag_start_pos
            self._drag_moved_px += math.hypot(delta.x(), delta.y())
            self._pan += delta
            self._drag_start_pos = pos
            self.update()
            return

        hovered = self._find_node_at_widget_pos(pos, hit_radius=13.0)
        if hovered != self._hovered_node_id:
            self._hovered_node_id = hovered
            self.setCursor(
                Qt.CursorShape.PointingHandCursor if hovered is not None else Qt.CursorShape.OpenHandCursor
            )
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._is_dragging_node:
                nid = self._dragging_node_id
                moved = self._drag_moved_px
                self._is_dragging_node = False
                self._dragging_node_id = None

                if moved < 5.0 and nid is not None:
                    self._selected_node_id = nid
                    self.node_left_clicked.emit(nid)
                elif moved >= 5.0 and nid is not None:
                    map_x, map_y, norm_x, norm_y = self._widget_to_node_coords(event.position())
                    self._save_node_position(nid, map_x, map_y, norm_x, norm_y)

            elif self._is_dragging:
                moved = self._drag_moved_px
                self._is_dragging = False
                if moved < 5.0:
                    hit_id = self._find_node_at_widget_pos(event.position(), hit_radius=13.0)
                    if hit_id is None:
                        map_x, map_y, norm_x, norm_y = self._widget_to_node_coords(event.position())
                        self._create_node_at_coords(map_x, map_y, norm_x, norm_y)
                    else:
                        self._selected_node_id = hit_id
                        self.node_left_clicked.emit(hit_id)

            hovered = self._find_node_at_widget_pos(event.position(), hit_radius=13.0)
            self.setCursor(
                Qt.CursorShape.PointingHandCursor if hovered is not None else Qt.CursorShape.OpenHandCursor
            )
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._is_dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(event)


    # -------------------------------------------------------------------------
    # Painting (Grayscale Map + Anti-aliased Node Badges + Outlined Numbers)
    # -------------------------------------------------------------------------
    NODE_STATE_COLORS: Dict[str, str] = {
        "ready": "#22c55e",
        "resting": "#f59e0b",
        "out_of_route": "#64748b",
    }

    def _classify_node(self, node: Dict[str, Any], now: Optional[float] = None) -> str:
        """
        Returns 'ready' (green #22c55e), 'resting' (amber #f59e0b),
        or 'out_of_route' (gray #64748b).

        P2-7 (K-02/K-06): gri kova "rota dışı / toplanamaz" demektir —
        efsane metni ("Başarısız") bu kovayı anlatmıyordu. `failed`
        durumu P1-6 `effective_node_state` ile aynı öncelikle tanınır
        (karantina > başarısız > dinlenen > aktif): `state == "failed"`
        (ya da "quarantined") ya da `consecutive_failures > 0` olan
        düğüm toplanamaz, gri kovaya girer; eskiden buradan geçip
        yeşil/turuncu (hazır/dinleniyor) çiziliyordu.
        """
        if now is None:
            now = time.time()
        nid = int(node.get("id", 0))
        explicit_status = str(node.get("status", node.get("state", "")) or "").strip().lower()
        if node.get("in_route") is False:
            in_route = False
        elif self._route_node_ids:
            in_route = nid in self._route_node_ids
        else:
            in_route = bool(node.get("in_route", True))

        if not in_route or explicit_status in ("out", "out_of_route", "gray", "disabled", "inactive"):
            return "out_of_route"

        # P2-7: `failed` tanınsın — başarısız düğüm hazır/dinleniyor
        # sayılmaz; P1-6 önceliğiyle (failed > resting) gri kovaya girer.
        if explicit_status in ("failed", "fail", "error", "quarantined"):
            return "out_of_route"
        try:
            if int(node.get("consecutive_failures", 0) or 0) > 0:
                return "out_of_route"
        except (TypeError, ValueError):
            pass

        if explicit_status in ("resting", "cooldown", "amber"):
            return "resting"

        last_visited = float(
            node.get("last_visited", node.get("last_visited_timestamp", 0)) or 0
        )
        rest_dur = float(node.get("rest_duration_sec", self._rest_duration_sec) or self._rest_duration_sec)
        if last_visited > 0 and (now - last_visited) < rest_dur:
            return "resting"

        return "ready"

    def get_node_state_color(self, node: Dict[str, Any], now: Optional[float] = None) -> str:
        """Returns the hex color string (#22c55e, #f59e0b, or #64748b) for the node's current state."""
        state = self._classify_node(node, now=now)
        return self.NODE_STATE_COLORS.get(state, "#22c55e")

    def _get_sequential_route_nodes(self) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Returns an ordered list of (seq_number, node_dict) for nodes in the active route.
        Preserves the sequential order from _route_node_ids_list, or sorts by seq_index.
        When a selective gather material filter is active, returns matching nodes in proximity order.
        """
        if not self._nodes:
            return []

        # When selective gather material filter is active, order matching nodes by proximity
        if self._material_filter:
            ordered_prox = self.calculate_proximity_route()
            if ordered_prox:
                return [(idx + 1, n) for idx, n in enumerate(ordered_prox)]
            return []

        node_map: Dict[int, Dict[str, Any]] = {}
        for n in self._nodes:
            if isinstance(n, dict):
                try:
                    node_map[int(n.get("id", 0))] = n
                except Exception:
                    continue

        ordered: List[Tuple[int, Dict[str, Any]]] = []
        if self._route_node_ids_list:
            seq = 1
            for nid in self._route_node_ids_list:
                if nid in node_map:
                    ordered.append((seq, node_map[nid]))
                    seq += 1
        else:
            active_nodes = [
                n for n in self._nodes
                if isinstance(n, dict) and (n.get("in_route", False) or int(n.get("id", 0)) in self._route_node_ids)
            ]
            active_nodes.sort(key=lambda item: int(item.get("seq_index", 0)))
            for idx, n in enumerate(active_nodes):
                ordered.append((idx + 1, n))

        return ordered

    def _draw_sequential_route(
        self,
        painter: QPainter,
        ordered_nodes: List[Tuple[int, Dict[str, Any]]],
    ) -> None:
        """
        Draws visual sequential route lines, glowing path corridors, directional
        arrows (1 ➔ 2 ➔ 3), and step sequence tags between active route nodes.
        """
        n_nodes = len(ordered_nodes)
        if n_nodes < 2:
            return

        closed = (n_nodes >= 3)
        count = n_nodes if closed else (n_nodes - 1)

        step_font = QFont("Segoe UI", 9)
        step_font.setWeight(QFont.Weight.Bold)

        for i in range(count):
            seq_a, node_a = ordered_nodes[i]
            seq_b, node_b = ordered_nodes[(i + 1) % n_nodes]
            is_closing = (i == n_nodes - 1)

            try:
                nx_a = float(node_a.get("map_x", node_a.get("x", 0.0)))
                ny_a = float(node_a.get("map_y", node_a.get("y", 0.0)))
                p1 = self._map_to_widget(self._node_to_map_coords(
                    nx_a, ny_a, norm_x=node_a.get("norm_x"), norm_y=node_a.get("norm_y"), node=node_a
                ))

                nx_b = float(node_b.get("map_x", node_b.get("x", 0.0)))
                ny_b = float(node_b.get("map_y", node_b.get("y", 0.0)))
                p2 = self._map_to_widget(self._node_to_map_coords(
                    nx_b, ny_b, norm_x=node_b.get("norm_x"), norm_y=node_b.get("norm_y"), node=node_b
                ))

                dx = p2.x() - p1.x()
                dy = p2.y() - p1.y()
                seg_len = math.hypot(dx, dy)
                if seg_len < 1.0:
                    continue

                ux = dx / seg_len
                uy = dy / seg_len

                if is_closing:
                    # Closing return segment: dashed line with subtle return arrow
                    dash_pen = QPen(
                        QColor(56, 189, 248, 140),
                        1.8,
                        Qt.PenStyle.DashLine,
                        Qt.PenCapStyle.RoundCap,
                    )
                    painter.setPen(dash_pen)
                    painter.drawLine(p1, p2)

                    if seg_len >= 32.0:
                        mid = QPointF((p1.x() + p2.x()) * 0.5, (p1.y() + p2.y()) * 0.5)
                        self._draw_chevron_arrow(painter, mid, ux, uy, QColor(56, 189, 248, 180), size=7.5)
                        if seg_len >= 60.0:
                            self._draw_step_badge(painter, QPointF(mid.x() - uy * 10.0, mid.y() + ux * 10.0), f"{seq_a} ➔ 1 ↺", step_font)
                    continue

                is_prox = bool(self._material_filter)

                # Soft cyan/gold corridor glow
                corridor_glow = QPen(
                    QColor(250, 204, 21, 80) if is_prox else QColor(14, 165, 233, 60),
                    8.0,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
                painter.setPen(corridor_glow)
                painter.drawLine(p1, p2)

                # Main crisp route line (#facc15 for proximity, #22d3ee for default)
                main_pen = QPen(
                    QColor("#facc15") if is_prox else QColor("#22d3ee"),
                    3.0 if is_prox else 2.8,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
                painter.setPen(main_pen)
                painter.drawLine(p1, p2)

                # Directional chevron arrows along segment
                if seg_len >= 90.0:
                    arrow_positions = (0.35, 0.70)
                elif seg_len >= 24.0:
                    arrow_positions = (0.55,)
                else:
                    arrow_positions = ()

                arrow_color = QColor("#fde047") if is_prox else QColor("#38bdf8")
                for ratio in arrow_positions:
                    ax = p1.x() + dx * ratio
                    ay = p1.y() + dy * ratio
                    self._draw_chevron_arrow(
                        painter,
                        QPointF(ax, ay),
                        ux,
                        uy,
                        arrow_color,
                        size=9.5 if is_prox else 9.0,
                    )

                # Sequence step tag (e.g. "1 ➔ 2" or "#1 ➔ #2") along route line
                if seg_len >= 42.0:
                    mid_x = (p1.x() + p2.x()) * 0.5
                    mid_y = (p1.y() + p2.y()) * 0.5
                    # Offset slightly perpendicular to route line
                    px, py = -uy, ux
                    bx = mid_x + px * 10.0
                    by = mid_y + py * 10.0
                    tag_str = f"#{seq_a} ➔ #{seq_b}" if is_prox else f"{seq_a} ➔ {seq_b}"
                    self._draw_step_badge(painter, QPointF(bx, by), tag_str, step_font)

            except Exception as exc:
                logger.debug("Error drawing sequential route segment: %s", exc)

    @staticmethod
    def _draw_chevron_arrow(
        painter: QPainter,
        tip_pt: QPointF,
        ux: float,
        uy: float,
        fill_color: QColor,
        size: float = 9.0,
    ) -> None:
        """Draws filled directional arrow in direction of unit vector (ux, uy)."""
        px, py = -uy, ux
        half_t = size * 0.5
        wing = size * 0.62

        tip = QPointF(tip_pt.x() + ux * half_t, tip_pt.y() + uy * half_t)
        left = QPointF(
            tip_pt.x() - ux * half_t + px * wing,
            tip_pt.y() - uy * half_t + py * wing,
        )
        indent = QPointF(
            tip_pt.x() - ux * (half_t * 0.35),
            tip_pt.y() - uy * (half_t * 0.35),
        )
        right = QPointF(
            tip_pt.x() - ux * half_t - px * wing,
            tip_pt.y() - uy * half_t - py * wing,
        )

        poly = QPolygonF([tip, left, indent, right])
        painter.setPen(QPen(QColor("#082f49"), 1.5))
        painter.setBrush(QBrush(fill_color))
        painter.drawPolygon(poly)

    @staticmethod
    def _draw_step_badge(
        painter: QPainter,
        center: QPointF,
        text: str,
        font: QFont,
    ) -> None:
        """Draws sequence step tag (e.g. 1 ➔ 2) along route segment."""
        painter.save()
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = max(18.0, float(fm.horizontalAdvance(text)))
        th = 15.0
        rect = QRectF(center.x() - tw * 0.5 - 3.0, center.y() - th * 0.5, tw + 6.0, th)
        painter.setPen(QPen(QColor("#0284c7"), 1.0))
        painter.setBrush(QBrush(QColor(15, 23, 42, 215)))
        painter.drawRoundedRect(rect, 3.5, 3.5)
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

    @staticmethod
    def _draw_proximity_order_badge(
        painter: QPainter,
        x: float,
        y: float,
        rank: int,
        font: QFont,
    ) -> None:
        """Draws a prominent gold/cyan proximity badge (#1, #2...) for matching selective gather nodes."""
        painter.save()
        painter.setFont(font)
        badge_text = f"#{rank}"
        fm = painter.fontMetrics()
        bw = max(24.0, float(fm.horizontalAdvance(badge_text)) + 10.0)
        bh = 16.0
        rect = QRectF(x, y - bh * 0.5, bw, bh)

        painter.setPen(QPen(QColor("#facc15"), 1.5))
        painter.setBrush(QBrush(QColor(15, 23, 42, 235)))
        painter.drawRoundedRect(rect, 3.5, 3.5)

        painter.setPen(QColor("#fde047"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, badge_text)
        painter.restore()

    def resolve_node_patch_path(self, node: Dict[str, Any], region: Optional[str] = None) -> str:
        """
        Resolves the file path of the 80x80 grayscale patch for the given node.
        Prioritizes:
        1. routes/<region>/patches/node_<id>.png
        2. patch_path or patch_file attribute on node
        3. routes/patches/<region>_node_<id>.png
        4. routes/patches/node_<id>.png
        """
        nid = int(node.get("id", 0)) if isinstance(node, dict) else getattr(node, "id", 0)
        reg = region
        if not reg:
            reg = node.get("region") if isinstance(node, dict) else getattr(node, "region", None)
        if not reg:
            parent = self.parent()
            reg = getattr(parent, "_region_name", "Altgard") if parent else "Altgard"
        reg = str(reg or "Altgard")

        # Canonical standard path (P3-6: kök-bağımlı; eski CWD'ye göreli
        # os.path.join("routes", ...) bot başka dizinden başlatılınca bulamazdı).
        primary_path = str(project_path("routes", reg, "patches", f"node_{nid}.png"))
        if os.path.isfile(primary_path):
            return primary_path

        # Explicit node patch_path / patch_file. Saklanan değer repo-göreli
        # olabilir ("routes/...", kalıcı JSON'da makineden bağımsız kalsın
        # diye böyle yazılır); okurken köke göre çözülür, mutlaksa aynen geçer.
        raw_path = (node.get("patch_path") if isinstance(node, dict) else getattr(node, "patch_path", "")) or \
                   (node.get("patch_file") if isinstance(node, dict) else getattr(node, "patch_file", ""))
        if raw_path:
            raw_resolved = str(resolve_project_path(str(raw_path)))
            if os.path.isfile(raw_resolved):
                return raw_resolved

        for cand in (
            str(project_path("routes", "patches", f"{reg}_node_{nid}.png")),
            str(project_path("routes", "patches", f"node_{nid}.png")),
            str(project_path("routes", reg, f"node_{nid}.png")),
        ):
            if os.path.isfile(cand):
                return cand

        return primary_path

    def get_node_patch_pixmap(self, node: Dict[str, Any], size: int = 64) -> QPixmap:
        """
        Loads and scales the node's 80x80 patch image, or generates a clean fallback thumbnail.
        Cached in self._patch_cache for smooth rendering.
        """
        nid = int(node.get("id", 0)) if isinstance(node, dict) else getattr(node, "id", 0)
        if nid in self._patch_cache:
            return self._patch_cache[nid]

        path = self.resolve_node_patch_path(node)
        pix: Optional[QPixmap] = None
        if os.path.isfile(path):
            try:
                img = QImage(path)
                if not img.isNull():
                    pix = QPixmap.fromImage(img).scaled(
                        size, size,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
            except Exception as exc:
                logger.debug("Error loading node patch '%s': %s", path, exc)

        if pix is None or pix.isNull():
            fallback = QImage(size, size, QImage.Format.Format_RGB32)
            fallback.fill(QColor("#1e293b"))
            p = QPainter(fallback)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setPen(QPen(QColor("#334155"), 1))
            for g in range(0, size, 16):
                p.drawLine(g, 0, g, size)
                p.drawLine(0, g, size, g)

            rtype = str(node.get("resource_type", "") if isinstance(node, dict) else getattr(node, "resource_type", ""))
            color = QColor("#22c55e") if ("Herb" in rtype or "Aria" in rtype) else (
                QColor("#f59e0b") if ("Ore" in rtype or "Orichalcum" in rtype) else QColor("#38bdf8")
            )
            p.setPen(QPen(color, 2))
            p.setBrush(QBrush(QColor(15, 23, 42, 200)))
            p.drawRoundedRect(QRectF(5, 5, size - 10, size - 10), 4, 4)

            font = QFont("Segoe UI", 10, QFont.Weight.Bold)
            p.setFont(font)
            p.setPen(QColor("#ffffff"))
            p.drawText(fallback.rect(), Qt.AlignmentFlag.AlignCenter, f"#{nid}")
            p.end()
            pix = QPixmap.fromImage(fallback)

        self._patch_cache[nid] = pix
        return pix

    def _draw_node_patch_preview_card(
        self,
        painter: QPainter,
        node: Dict[str, Any],
        pin_pt: QPointF,
    ) -> None:
        """Draws floating card showing routes/<region>/patches/node_<id>.png preview and metadata."""
        nid = int(node.get("id", 0))
        thumb_size = 64
        thumb_pix = self.get_node_patch_pixmap(node, size=thumb_size)

        card_w = 280.0
        card_h = 92.0

        card_x = pin_pt.x() + 18.0
        card_y = pin_pt.y() - card_h - 10.0

        if card_x + card_w > self.width() - 10.0:
            card_x = pin_pt.x() - card_w - 18.0
        if card_x < 10.0:
            card_x = 10.0
        if card_y < 10.0:
            card_y = pin_pt.y() + 18.0
        if card_y + card_h > self.height() - 36.0:
            card_y = self.height() - card_h - 36.0

        card_rect = QRectF(card_x, card_y, card_w, card_h)

        painter.save()
        # Card body with cyan accent border
        painter.setPen(QPen(QColor("#0284c7"), 1.5))
        painter.setBrush(QBrush(QColor(15, 23, 42, 240)))
        painter.drawRoundedRect(card_rect, 6.0, 6.0)

        # Thumbnail frame
        thumb_rect = QRectF(card_x + 8.0, card_y + 11.0, float(thumb_size), float(thumb_size))
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.setBrush(QBrush(QColor("#090d16")))
        painter.drawRoundedRect(thumb_rect, 4.0, 4.0)
        painter.drawPixmap(thumb_rect.toRect(), thumb_pix)

        # Labels
        tx = card_x + 8.0 + thumb_size + 10.0
        ty = card_y + 18.0

        title_font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor("#38bdf8"))
        seq_idx = node.get("seq_index")
        seq_text = f" [#{seq_idx}]" if seq_idx else ""
        painter.drawText(QPointF(tx, ty), f"Nokta #{nid}{seq_text}")

        info_font = QFont("Segoe UI", 9)
        painter.setFont(info_font)
        painter.setPen(QColor("#fde047"))
        rtype = str(node.get("resource_type", "Orichalcum Ore"))
        painter.drawText(QPointF(tx, ty + 16.0), rtype)

        painter.setPen(QColor("#94a3b8"))
        mx = node.get("map_x", 0)
        my = node.get("map_y", 0)
        painter.drawText(QPointF(tx, ty + 32.0), f"Konum: ({mx}, {my})")

        badge_str = self._format_harvest_badge(node)
        state_str = str(node.get("state", "active")).title()
        painter.setPen(QColor("#4ade80") if badge_str.endswith("x") else QColor("#67e8f9"))
        painter.drawText(QPointF(tx, ty + 48.0), f"Hasat: {badge_str} · {state_str}")

        painter.restore()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # Fill dark background outside map bounds
        painter.fillRect(self.rect(), QColor("#14171c"))

        # 1. Draw Grayscale Region Map
        if self._base_pixmap is not None and not self._base_pixmap.isNull():
            target_rect = QRectF(
                self._pan.x(),
                self._pan.y(),
                self._base_pixmap.width() * self._zoom,
                self._base_pixmap.height() * self._zoom
            )
            painter.drawPixmap(target_rect, self._base_pixmap, QRectF(self._base_pixmap.rect()))

        # 2. Draw Visual Sequential Route Lines & Directional Arrows (1 ➔ 2 ➔ 3)
        ordered_route_nodes = self._get_sequential_route_nodes()
        self._draw_sequential_route(painter, ordered_route_nodes)
        seq_map: Dict[int, int] = {
            int(n.get("id", 0)): s_idx for s_idx, n in ordered_route_nodes
        }

        # 3. Draw Player Position Marker (Blue Arrow / Dot #3b82f6) if available
        if self._player_pos is not None:
            px, py, angle_deg = self._player_pos
            w_pt = self._map_to_widget(self._node_to_map_coords(px, py))
            self._draw_player_marker(painter, w_pt, angle_deg)

        # 4. Draw All Nodes (Green #22c55e, Amber #f59e0b, Gray #64748b)
        #    + Cyan/Emerald dual ring for minimap-auto-recorded nodes (source_type == "minimap_auto")
        #    + Harvest status badge (0/3, 1/3, 2/3 or 3x) next to each node pin
        now = time.time()
        color_map = {
            "ready": QColor("#22c55e"),        # Green in the route
            "resting": QColor("#f59e0b"),      # Amber resting
            "out_of_route": QColor("#64748b")  # Gray out of route
        }

        font = QFont("Segoe UI", 10)
        font.setWeight(QFont.Weight.Bold)
        badge_font = QFont("Segoe UI", 9)
        badge_font.setWeight(QFont.Weight.Bold)

        is_filter_active = bool(self._material_filter)

        for node in self._nodes:
            try:
                nid = int(node.get("id", 0))
                nx = float(node.get("map_x", node.get("x", 0)))
                ny = float(node.get("map_y", node.get("y", 0)))
                norm_x = node.get("norm_x")
                norm_y = node.get("norm_y")
                w_pt = self._map_to_widget(self._node_to_map_coords(nx, ny, norm_x=norm_x, norm_y=norm_y, node=node))

                is_matching = node_matches_material(node, self._material_filter) if is_filter_active else True

                # Dim non-matching nodes to 0.20 opacity
                if is_filter_active and not is_matching:
                    painter.save()
                    painter.setOpacity(0.20)

                state = self._classify_node(node, now)
                fill_color = color_map.get(state, QColor("#22c55e"))
                is_hovered = (nid == self._hovered_node_id)
                is_selected = (nid == self._selected_node_id)
                is_minimap_auto = self._is_minimap_auto_node(node)

                radius = 7.5 if is_hovered else 6.0
                border_width = 2.2 if is_hovered else 1.8

                # Distinct cyan/emerald dual ring for minimap-auto-recorded nodes (source_type == "minimap_auto")
                if is_minimap_auto:
                    self._draw_minimap_auto_dual_ring(painter, w_pt, radius, is_hovered)

                # Selected halo
                if is_selected and not is_hovered:
                    painter.setPen(QPen(QColor("#38bdf8"), 2.2, Qt.PenStyle.DashLine))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(w_pt, radius + 3.5, radius + 3.5)

                # Proximity match highlight with glowing double ring (gold/cyan)
                if is_filter_active and is_matching:
                    glow_r = radius + (5.5 if is_hovered else 4.2)
                    inner_glow_r = radius + (3.0 if is_hovered else 2.2)
                    painter.setPen(QPen(QColor("#facc15"), 2.4))
                    painter.setBrush(QBrush(QColor(250, 204, 21, 40)))
                    painter.drawEllipse(w_pt, glow_r, glow_r)
                    painter.setPen(QPen(QColor("#00ffff"), 1.8))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(w_pt, inner_glow_r, inner_glow_r)

                # Circle with crisp black border (matches frame_24s.jpg)
                if is_hovered:
                    painter.setPen(QPen(QColor("#ffffff"), 3.2))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(w_pt, radius + 1.5, radius + 1.5)

                painter.setPen(QPen(QColor("#000000"), border_width))
                painter.setBrush(QBrush(fill_color))
                painter.drawEllipse(w_pt, radius, radius)

                # Bold white node number with sequence info and dark outline/shadow to the right
                painter.setFont(font)
                seq_num = seq_map.get(nid)
                if seq_num is not None:
                    label_str = f"#{nid} [#{seq_num}]"
                else:
                    label_str = f"#{nid}"

                ring_pad = 5.5 if is_minimap_auto else 0.0
                tx = w_pt.x() + radius + 3.5 + ring_pad
                ty = w_pt.y() + 4.0

                self._draw_outlined_text(painter, tx, ty, label_str, font)

                # Harvest status badge (0/3, 1/3, 2/3 or 3x) next to the node pin & number
                badge_text = self._format_harvest_badge(node)
                fm = painter.fontMetrics()
                id_width = float(fm.horizontalAdvance(label_str))
                badge_x = tx + id_width + 5.0
                badge_y = w_pt.y()
                self._draw_harvest_status_badge(
                    painter=painter,
                    x=badge_x,
                    y=badge_y,
                    badge_text=badge_text,
                    font=badge_font,
                    is_minimap_auto=is_minimap_auto,
                    state=state,
                )

                # Prominent proximity order badge #1, #2, #3... for matching nodes
                if is_filter_active and is_matching and seq_num is not None:
                    badge_fm = painter.fontMetrics()
                    hb_width = float(badge_fm.horizontalAdvance(badge_text))
                    prox_badge_x = badge_x + hb_width + 12.0
                    self._draw_proximity_order_badge(
                        painter=painter,
                        x=prox_badge_x,
                        y=badge_y,
                        rank=seq_num,
                        font=badge_font,
                    )

                if is_filter_active and not is_matching:
                    painter.restore()
            except Exception as exc:
                logger.debug("Error drawing node on map: %s", exc)

        # 4b. Draw Hovered Node Patch Preview Card (routes/<region>/patches/node_<id>.png)
        if self._hovered_node_id is not None and not self._is_dragging and not self._is_dragging_node:
            hovered_node = next((n for n in self._nodes if int(n.get("id", -1)) == self._hovered_node_id), None)
            if hovered_node is not None:
                try:
                    nx = float(hovered_node.get("map_x", hovered_node.get("x", 0)))
                    ny = float(hovered_node.get("map_y", hovered_node.get("y", 0)))
                    pin_pt = self._map_to_widget(self._node_to_map_coords(
                        nx, ny,
                        norm_x=hovered_node.get("norm_x"),
                        norm_y=hovered_node.get("norm_y"),
                        node=hovered_node
                    ))
                    self._draw_node_patch_preview_card(painter, hovered_node, pin_pt)
                except Exception as exc:
                    logger.debug("Error drawing node patch preview card: %s", exc)

        # 5. Draw Map Legend (Pin states + Minimap-Auto Cyan/Emerald Dual Ring + Harvest Status Badge)
        self._draw_map_legend(painter)

        painter.end()

    @staticmethod
    def _is_minimap_auto_node(node: Dict[str, Any]) -> bool:
        """Returns True if the node was auto-recorded from the top-right minimap."""
        if not isinstance(node, dict):
            return False
        src = str(node.get("source_type", node.get("source", ""))).strip().lower()
        if src == "minimap_auto":
            return True
        return bool(node.get("minimap_auto", False))

    @staticmethod
    def _format_harvest_badge(node: Dict[str, Any]) -> str:
        """
        Formats the node's harvest status badge ('0/3', '1/3', '2/3' or '3x', or '6x'/'9x' for stacked nodes).
        """
        if not isinstance(node, dict):
            return "0/3"

        explicit_badge = node.get("harvest_badge") or node.get("harvest_status")
        if explicit_badge is not None and str(explicit_badge).strip():
            return str(explicit_badge).strip()

        max_h = max(1, int(node.get("max_harvests", node.get("stack_count", 3)) or 3))

        if "harvest_count" in node or "current_harvests" in node:
            cur_h = max(0, int(node.get("harvest_count", node.get("current_harvests", 0)) or 0))
            if str(node.get("badge_mode", "")).strip().lower() == "stack" or cur_h >= max_h:
                return f"{max_h}x"
            return f"{cur_h}/{max_h}"

        if "stack_badge" in node and str(node.get("stack_badge", "")).strip():
            return str(node.get("stack_badge")).strip()

        if "stack_count" in node and "visit_count" not in node and "gathered_count" not in node:
            return f"{max_h}x"

        visits = max(0, int(node.get("visit_count", node.get("gathered_count", 0)) or 0))
        cycle = visits % (max_h + 1)
        if cycle >= max_h:
            return f"{max_h}x"
        return f"{cycle}/{max_h}"

    @staticmethod
    def _draw_minimap_auto_dual_ring(
        painter: QPainter,
        center: QPointF,
        base_radius: float,
        is_hovered: bool = False,
    ) -> None:
        """
        Draws a distinct cyan/emerald dual ring around minimap-auto-recorded nodes
        (source_type == 'minimap_auto') so the user can visually identify them on the route map.
        """
        outer_r = base_radius + (6.2 if is_hovered else 5.2)
        inner_r = base_radius + (3.4 if is_hovered else 2.6)

        # Soft cyan-emerald halo glow
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(6, 182, 212, 48)))
        painter.drawEllipse(center, outer_r + 2.4, outer_r + 2.4)

        # Outer bright cyan ring (#22d3ee)
        outer_pen = QPen(QColor("#22d3ee"), 2.0, Qt.PenStyle.SolidLine)
        painter.setPen(outer_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, outer_r, outer_r)

        # Inner bright emerald ring (#10b981)
        inner_pen = QPen(QColor("#10b981"), 2.0, Qt.PenStyle.SolidLine)
        painter.setPen(inner_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, inner_r, inner_r)

    @staticmethod
    def _draw_harvest_status_badge(
        painter: QPainter,
        x: float,
        y: float,
        badge_text: str,
        font: QFont,
        is_minimap_auto: bool = False,
        state: str = "ready",
    ) -> None:
        """
        Draws the harvest status badge ('0/3', '1/3', '2/3' or '3x') as a compact pill next to the node pin.
        """
        painter.save()
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = max(18.0, float(fm.horizontalAdvance(badge_text)))
        pad_x = 4.5
        pill_w = tw + pad_x * 2.0
        pill_h = 16.0
        pill_rect = QRectF(x, y - pill_h * 0.5 + 0.5, pill_w, pill_h)

        if is_minimap_auto:
            bg_color = QColor(6, 44, 48, 230)
            border_color = QColor("#22d3ee") if not badge_text.endswith("x") else QColor("#10b981")
            text_color = QColor("#34d399") if badge_text.endswith("x") else QColor("#67e8f9")
        elif state == "resting":
            bg_color = QColor(45, 27, 8, 225)
            border_color = QColor("#f59e0b")
            text_color = QColor("#fde68a")
        elif state == "out_of_route":
            bg_color = QColor(23, 30, 42, 215)
            border_color = QColor("#64748b")
            text_color = QColor("#cbd5e1")
        else:
            bg_color = QColor(12, 34, 24, 225)
            border_color = QColor("#22c55e")
            text_color = QColor("#4ade80") if badge_text.endswith("x") else QColor("#e2e8f0")

        painter.setPen(QPen(border_color, 1.3))
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(pill_rect, 4.0, 4.0)

        painter.setPen(text_color)
        painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, badge_text)
        painter.restore()

    def _draw_map_legend(self, painter: QPainter) -> None:
        """Draws a bottom Turkish legend explaining node colors, minimap_auto dual rings, and 0/3..3x badges."""
        if self.height() < 120 or self.width() < 260:
            return
        legend_w = min(float(self.width() - 20), 920.0)
        legend_rect = QRectF(10.0, float(self.height() - 38), legend_w, 28.0)
        painter.save()
        painter.setPen(QPen(QColor(45, 55, 72, 200), 1.0))
        painter.setBrush(QBrush(QColor(15, 20, 28, 220)))
        painter.drawRoundedRect(legend_rect, 5.0, 5.0)

        l_font = QFont("Segoe UI", 9)
        l_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(l_font)
        painter.setPen(QColor("#e2e8f0"))
        legend_str = (
            "  ● Aktif   |   ● Dinleniyor (5dk)   |   ● Rota dışı   |   "
            "◎ Minimap/3B Oto-Kayıt   |   [0/3 · 1/3 · 2/3 · 3x]: Hasat Rozeti   |   "
            "Sol Tık: Seç / Boş Alana Nokta Ekle | Sağ Tık: Menü (Shift+Sağ Tık: Sil) | Sürükle: Taşı"
        )
        painter.drawText(
            legend_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            legend_str,
        )
        painter.restore()

    @staticmethod
    def _draw_outlined_text(
        painter: QPainter,
        x: float,
        y: float,
        text: str,
        font: QFont
    ) -> None:
        """
        Draws crisp white bold text with a black stroke/outline for readability
        against any grayscale terrain (matching frame_24s.jpg).
        """
        path = QPainterPath()
        path.addText(QPointF(x, y), font, text)
        painter.setPen(
            QPen(
                QColor(0, 0, 0, 235),
                3.0,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin
            )
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawPath(path)

    @staticmethod
    def _draw_player_marker(painter: QPainter, pt: QPointF, angle_deg: float = 0.0) -> None:
        """Draws the player's current position as a blue directional arrow/dot (#3b82f6)."""
        painter.save()
        painter.translate(pt)
        painter.rotate(angle_deg)

        # Outer glow ring
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(59, 130, 246, 80)))
        painter.drawEllipse(QPointF(0, 0), 11.0, 11.0)

        # Directional arrow polygon (#3b82f6 with black/white border)
        arrow = QPolygonF([
            QPointF(0.0, -9.0),
            QPointF(6.5, 7.0),
            QPointF(0.0, 3.5),
            QPointF(-6.5, 7.0),
        ])
        painter.setPen(QPen(QColor("#000000"), 2.0))
        painter.setBrush(QBrush(QColor("#3b82f6")))
        painter.drawPolygon(arrow)

        painter.restore()


class MapViewerWindow(QDialog):
    """
    Rota ve Harita Görüntüleyici Penceresi (Gatherion 1.5.2 Uyumlu Türkçe Arayüz).
    Harita üzerindeki tüm noktaları, hasat durum rozetlerini (0/3, 1/3, 2/3, 3x) ve
    Minimap/3B Oto-Kayıt (source_type == 'minimap_auto') çift halkalı noktaları gösterir.
    """
    node_toggled = pyqtSignal(int)
    node_deleted = pyqtSignal(int)
    node_added = pyqtSignal(dict)
    node_moved = pyqtSignal(int, float, float)

    def __init__(
        self,
        route_manager: Optional[Any] = None,
        region_name: Optional[str] = None,
        route_name: Optional[str] = None,
        map_image_path: Optional[str] = None,
        rest_duration_sec: float = 300.0,
        player_pos: Optional[Tuple[float, float]] = None,
        parent: Optional[QWidget] = None,
        **kwargs: Any,
    ):
        super().__init__(parent)
        if region_name is None and "region" in kwargs:
            region_name = kwargs.pop("region")
        if route_name is None and "route" in kwargs:
            route_name = kwargs.pop("route")
        self.routes = route_manager
        self._region_name = region_name or self._resolve_active_region()
        self._route_name = route_name or self._resolve_active_route()
        self._explicit_map_path = map_image_path
        # Görev 4: bölgeye ait HD harita görseli yoksa bölge adı burada tutulur
        # ("bu bölge için harita yok" durumu; Altgard görseli kullanılmaz).
        self._map_missing_for_region: str = ""
        self._rest_duration_sec = float(rest_duration_sec)
        # Normalise to the (x, y, angle_deg) triple the canvas uses so the
        # heading survives reloads instead of being dropped.
        if player_pos is not None and len(player_pos) == 2:
            player_pos = (float(player_pos[0]), float(player_pos[1]), 0.0)
        elif player_pos is not None:
            player_pos = (float(player_pos[0]), float(player_pos[1]), float(player_pos[2]))
        self._player_pos = player_pos

        self._init_ui()
        self.reload_map_and_nodes()

    def _resolve_active_region(self) -> str:
        try:
            if self.routes is not None:
                if hasattr(self.routes, "active_region") and self.routes.active_region:
                    return str(self.routes.active_region)
                if hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                    reg = self.routes.data.get("active_region")
                    if reg:
                        return str(reg)
        except Exception as exc:
            logger.debug("Could not resolve active_region: %s", exc)
        return "Altgard"

    def _resolve_active_route(self) -> str:
        try:
            if self.routes is not None:
                if hasattr(self.routes, "active_route") and self.routes.active_route:
                    return str(self.routes.active_route)
                if hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                    rte = self.routes.data.get("active_route")
                    if rte:
                        return str(rte)
        except Exception as exc:
            logger.debug("Could not resolve active_route: %s", exc)
        return "All"

    def _build_title_text(self) -> str:
        region_label = self._region_name or "Altgard"
        route_label = self._route_name or "All"
        title = (
            f"Rota ve Harita Görüntüleyici · Bölge: {region_label} | Rota: {route_label} · "
            "Aktif, Dinleniyor (5dk), Rota dışı, Minimap/3B Oto-Kayıt"
        )
        # Görev 4: bölge uyumsuzluğu varsa başlıkta görünür uyarı.
        try:
            getter = getattr(self.routes, "get_region_mismatch", None)
            mismatch = getter() if callable(getter) else None
        except Exception:
            mismatch = None
        if isinstance(mismatch, dict):
            title += (
                f" · ⚠ BÖLGE UYUMSUZLUĞU: {mismatch.get('detected_region')} ≠ "
                f"{mismatch.get('active_region')} — rota durduruldu"
            )
        return title

    def _init_ui(self) -> None:
        title_str = self._build_title_text()
        self.setWindowTitle(title_str)
        self.resize(980, 680)
        self.setStyleSheet("""
            QDialog {
                background-color: #14171c;
                color: #e5e7eb;
                border: 1px solid #2d323b;
                font-family: 'Segoe UI', sans-serif;
            }
            QWidget#MapTopHeader, QWidget#MapBottomFooter {
                background-color: #181c23;
                border-bottom: 1px solid #2d323b;
            }
            QLabel {
                color: #e5e7eb;
                font-size: 12px;
            }
            QPushButton {
                background-color: #0284c7;
                color: #ffffff;
                border: 1px solid #38bdf8;
                border-radius: 5px;
                padding: 7px 14px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0369a1;
            }
        """)

        root_layout = wrap_shell_tab_root_layout(self, "map")
        root_layout.setSpacing(0)

        # Üst Türkçe Bilgi & Kontrol Çubuğu
        self.header_bar = QWidget(self)
        self.header_bar.setObjectName("MapTopHeader")
        header_layout = QHBoxLayout(self.header_bar)
        header_layout.setContentsMargins(14, 8, 14, 8)
        header_layout.setSpacing(12)

        self.lbl_region_prefix = QLabel("Bölge:", self.header_bar)
        self.lbl_region_prefix.setStyleSheet("color: #94a3b8; font-weight: bold;")
        header_layout.addWidget(self.lbl_region_prefix)

        self.lbl_region_value = QLabel(str(self._region_name), self.header_bar)
        self.lbl_region_value.setStyleSheet("color: #f8fafc; font-weight: bold;")
        header_layout.addWidget(self.lbl_region_value)

        self.lbl_route_prefix = QLabel("Rota:", self.header_bar)
        self.lbl_route_prefix.setStyleSheet("color: #94a3b8; font-weight: bold;")
        header_layout.addWidget(self.lbl_route_prefix)

        self.lbl_route_value = QLabel(str(self._route_name), self.header_bar)
        self.lbl_route_value.setStyleSheet("color: #f8fafc; font-weight: bold;")
        header_layout.addWidget(self.lbl_route_value)

        header_layout.addStretch(1)

        # Durum Rozetleri (Aktif | Dinleniyor (5dk) | Rota dışı | Minimap/3B Oto-Kayıt)
        # P2-7: gri rozet "Başarısız" değil "Rota dışı" yazar — gri kova
        # _classify_node'da out_of_route'dur (rota dışı / başarısız /
        # karantinalı / toplanamaz); "Başarısız" etiketi yanıltıcıydı.
        self.lbl_badge_active = QLabel("● Aktif", self.header_bar)
        self.lbl_badge_active.setStyleSheet("color: #22c55e; font-weight: bold; font-size: 12px;")
        header_layout.addWidget(self.lbl_badge_active)

        self.lbl_badge_resting = QLabel("● Dinleniyor (5dk)", self.header_bar)
        self.lbl_badge_resting.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 12px;")
        header_layout.addWidget(self.lbl_badge_resting)

        self.lbl_badge_out_of_route = QLabel("● Rota dışı", self.header_bar)
        self.lbl_badge_out_of_route.setStyleSheet("color: #64748b; font-weight: bold; font-size: 12px;")
        header_layout.addWidget(self.lbl_badge_out_of_route)
        # Geriye dönük uyumluluk takma adı (eski ad: lbl_badge_failed).
        self.lbl_badge_failed = self.lbl_badge_out_of_route

        self.lbl_badge_minimap_auto = QLabel("◎ Minimap/3B Oto-Kayıt", self.header_bar)
        self.lbl_badge_minimap_auto.setStyleSheet("color: #d4a017; font-weight: bold; font-size: 12px;")
        header_layout.addWidget(self.lbl_badge_minimap_auto)

        self.btn_optimize_route = QPushButton("Sıralı Rota Optimize Et", self.header_bar)
        self.btn_optimize_route.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_optimize_route.setMinimumHeight(30)
        self.btn_optimize_route.clicked.connect(self.optimize_sequential_route)
        header_layout.addWidget(self.btn_optimize_route)

        # 🎯 Seç-Topla Filtresi Dropdown & 🧭 Yakınlık Rotası Çiz Butonu
        self.lbl_material_filter = QLabel("🎯 Seç-Topla Filtresi:", self.header_bar)
        self.lbl_material_filter.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 12px;")
        header_layout.addWidget(self.lbl_material_filter)

        self.combo_material_filter = QComboBox(self.header_bar)
        self.combo_material_filter.setObjectName("MapMaterialFilterCombo")
        self.combo_material_filter.setToolTip("🎯 Seç-Topla Filtresi: Toplanacak materyali seçin")
        self.combo_material_filter.setStyleSheet("""
            QComboBox {
                background-color: #1e293b;
                color: #ffffff;
                border: 1px solid #0284c7;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 12px;
                font-weight: bold;
                min-width: 160px;
            }
            QComboBox::drop-down { border: 0px; }
            QComboBox QAbstractItemView {
                background-color: #11151c;
                color: #ffffff;
                selection-background-color: #0284c7;
                selection-color: #ffffff;
            }
        """)
        # P3-7: başlık comboları tek geometri — HEADER_COMBO_MIN_HEIGHT.
        # (padding 5px 10px / min-width 160px aşağıdaki QSS'te bg moduyla aynı değerlerdedir.)
        self.combo_material_filter.setMinimumHeight(HEADER_COMBO_MIN_HEIGHT)
        if hasattr(self.combo_material_filter, "view") and self.combo_material_filter.view() is not None:
            view = self.combo_material_filter.view()
            palette = view.palette()
            palette.setColor(QPalette.ColorRole.Text, QColor("#ffffff"))
            palette.setColor(QPalette.ColorRole.WindowText, QColor("#ffffff"))
            palette.setColor(QPalette.ColorRole.Base, QColor("#11151c"))
            view.setPalette(palette)

        for label, data in [
            ("🎯 Tümü", "Tümü"),
            ("⛏️ Orichalcum Ore", "Orichalcum Ore"),
            ("🌸 Aria", "Aria"),
            ("🍒 Mela", "Mela"),
            ("🌿 Targena", "Targena"),
            ("💎 Gem / Crystal", "Gem / Crystal"),
            ("🪵 Tree / Log", "Tree / Log"),
        ]:
            self.combo_material_filter.addItem(label, userData=data)
        self.combo_material_filter.currentTextChanged.connect(self._on_material_filter_changed)
        header_layout.addWidget(self.combo_material_filter)

        self.btn_draw_proximity_route = QPushButton("🧭 Yakınlık Rotası Çiz", self.header_bar)
        self.btn_draw_proximity_route.setObjectName("MapDrawProximityRouteBtn")
        self.btn_draw_proximity_route.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_draw_proximity_route.setMinimumHeight(30)
        self.btn_draw_proximity_route.setToolTip("Seçilen materyalin noktalarını en yakın mesafeye göre sıralayıp rota çizer")
        self.btn_draw_proximity_route.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: #ffffff;
                border: 1px solid #38bdf8;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0369a1;
            }
        """)
        self.btn_draw_proximity_route.clicked.connect(self._on_draw_proximity_route_clicked)
        header_layout.addWidget(self.btn_draw_proximity_route)

        root_layout.addWidget(self.header_bar)

        # Setup background map controls & attributes
        self._background_mode: str = "auto"
        self.auto_cartographer: Optional[Any] = None
        self._last_auto_count: int = 0
        self._live_sync_timer = None

        if self.routes is not None:
            ac = getattr(self.routes, "auto_cartographer", None) or getattr(self.routes, "_auto_cartographer", None)
            if ac is not None:
                self.auto_cartographer = ac

        self.lbl_bg_mode = QLabel("Harita:", self.header_bar)
        self.lbl_bg_mode.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 12px;")
        self.combo_bg_mode = QComboBox(self.header_bar)
        # P3-7: materyal filtresi combosuyla aynı geometri (padding 5px 10px,
        # min-width 160px, min-height HEADER_COMBO_MIN_HEIGHT). Eskiden bu
        # combo 4px 8px padding taşıyor ve yükseklik almıyordu — iki kardeş
        # combo başlıkta farklı yükseklikte çiziliyordu.
        self.combo_bg_mode.setMinimumHeight(HEADER_COMBO_MIN_HEIGHT)
        self.combo_bg_mode.setStyleSheet("""
            QComboBox {
                background-color: #1e293b;
                color: #f1f5f9;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 12px;
                min-width: 160px;
            }
            QComboBox::drop-down { border: 0px; }
            QComboBox QAbstractItemView {
                background-color: #1e293b;
                color: #f1f5f9;
                selection-background-color: #0284c7;
            }
        """)
        self.combo_bg_mode.addItems([
            "Otomatik (HD/Kare/Mozaik)",
            "Altgard HD Harita",
            "Referans Kare (frame_18s)",
            "Mozaik Tuval (Yama)",
        ])
        self.combo_bg_mode.currentIndexChanged.connect(self._on_bg_mode_changed)
        header_layout = self.header_bar.layout()
        if header_layout is not None:
            header_layout.addWidget(self.lbl_bg_mode)
            header_layout.addWidget(self.combo_bg_mode)

        # Canvas
        self.canvas = _MapCanvas(self)
        self._canvas = self.canvas
        self._connect_canvas_signals()
        root_layout.addWidget(self.canvas, stretch=1)

        # Alt Türkçe Yardım Çubuğu
        self.footer_bar = QWidget(self)
        self.footer_bar.setObjectName("MapBottomFooter")
        footer_layout = QHBoxLayout(self.footer_bar)
        footer_layout.setContentsMargins(14, 6, 14, 6)
        self.lbl_footer_help = QLabel(
            "Sol Tık: Seç / Boş Alana Nokta Ekle | Sağ Tık: Menü (Shift+Sağ Tık: Sil) | Sürükle: Taşı",
            self.footer_bar,
        )
        self.lbl_footer_help.setStyleSheet("color: #94a3b8; font-size: 12px; font-weight: 600;")
        self.lbl_footer_help.setWordWrap(True)
        footer_layout.addWidget(self.lbl_footer_help)
        footer_layout.addStretch(1)
        root_layout.addWidget(self.footer_bar)

        # Live odometry & auto-mapping sync timer
        try:
            from PyQt6.QtCore import QTimer
            self._live_sync_timer = QTimer(self)
            self._live_sync_timer.setInterval(1000)
            self._live_sync_timer.timeout.connect(self.poll_live_updates)
            self._live_sync_timer.start()
        except Exception as exc:
            logger.debug("Could not start live sync timer: %s", exc)

    # -------------------------------------------------------------------------
    # Subregion Canonical Mapping
    # -------------------------------------------------------------------------
    # J (sweep): kendi kopya harita kaldırıldı — tek kaynak
    # core/route_manager.py SUBREGION_TO_CANONICAL. Bu sözlük yalnızca
    # onun küçük harfli türev görünümüdür (yeni kaynak eklenmez).
    # Eksik 5 map_viewer girdisi (Mumu Farmland, Dredgion, Calderon,
    # Gribbler Sanctuary, Basfelt Village) tek kaynağa taşındı.
    # canonical_region_name boş/None için "Altgard" varsayar (route_manager
    # DEFAULT_REGION "Verteron"dur — viewer varsayılanı korunur) ve alt
    # dize eşleşmesini de korur (örn. "Dredgion Crash Site (Elite)").
    SUBREGION_CANONICAL_MAP: Dict[str, str] = {
        str(k).strip().lower(): v for k, v in SUBREGION_TO_CANONICAL.items()
    }

    @classmethod
    def canonical_region_name(cls, name: Optional[str]) -> str:
        """
        Maps subregions ("Dredgion Crash Site", "Safe Haven", "Calderon Canyon", etc.)
        to canonical parent region "Altgard". If name is empty/None, defaults to "Altgard".
        """
        if not name:
            return "Altgard"
        raw = str(name).strip()
        lower = raw.lower()
        if lower in cls.SUBREGION_CANONICAL_MAP:
            return cls.SUBREGION_CANONICAL_MAP[lower]
        for sub, parent in cls.SUBREGION_CANONICAL_MAP.items():
            if sub in lower:
                return parent
        return raw

    # -------------------------------------------------------------------------
    # Background Map Mode Switching
    # -------------------------------------------------------------------------
    def set_background_mode(self, mode: str) -> None:
        """
        Seamlessly switches the background map mode between:
        - 'hd': bölgenin HD haritası (assets/maps/<bölge>.png; Altgard 2258x2400,
          Verteron 2303x2400). Bölgenin görseli yoksa Altgard haritası
          kullanılmaz, mozaik yedeğe düşülür.
        - 'reference_frame' or 'frame18': Cropped reference frame (video_frames/frame_18s.jpg)
        - 'mosaic': Constructed 80x80 node patch mosaic
        - 'auto': Automatic fallback (HD -> reference frame -> mosaic)
        """
        norm_mode = str(mode or "auto").strip().lower()
        if norm_mode in ("hd", "altgard_hd", "altgard", "assets/maps/altgard.png"):
            self._background_mode = "hd"
        elif norm_mode in ("reference_frame", "reference", "frame18", "frame", "video_frames/frame_18s.jpg"):
            self._background_mode = "frame18"
        elif norm_mode in ("mosaic", "mozaik", "patch", "patches"):
            self._background_mode = "mosaic"
        else:
            self._background_mode = "auto"

        if hasattr(self, "combo_bg_mode") and self.combo_bg_mode is not None:
            mode_map = {"auto": 0, "hd": 1, "frame18": 2, "mosaic": 3}
            idx = mode_map.get(self._background_mode, 0)
            if self.combo_bg_mode.currentIndex() != idx:
                self.combo_bg_mode.blockSignals(True)
                self.combo_bg_mode.setCurrentIndex(idx)
                self.combo_bg_mode.blockSignals(False)

        self.reload_map_and_nodes()

    def get_background_mode(self) -> str:
        return getattr(self, "_background_mode", "auto")

    def _on_bg_mode_changed(self, index: int) -> None:
        mode_lookup = {0: "auto", 1: "hd", 2: "frame18", 3: "mosaic"}
        mode = mode_lookup.get(index, "auto")
        self.set_background_mode(mode)

    # -------------------------------------------------------------------------
    # Grayscale Map Loading & Mosaic Fallback
    # -------------------------------------------------------------------------
    @staticmethod
    def _to_grayscale_pixmap(image: QImage) -> QPixmap:
        """Converts any QImage to a crisp 8-bit grayscale QPixmap matching frame_24s.jpg."""
        if image.isNull():
            return QPixmap()
        gray_img = image.convertToFormat(QImage.Format.Format_Grayscale8)
        # Convert back to RGB32 so QPainter draws colored circles on top if needed
        rgb_gray = gray_img.convertToFormat(QImage.Format.Format_RGB32)
        return QPixmap.fromImage(rgb_gray)

    def _load_frame_pixmap(self, path: str) -> Tuple[Optional[QPixmap], Tuple[int, int]]:
        try:
            img = QImage(path)
            if img.isNull():
                return None, (0, 0)
            rx, ry, rw, rh = _MapCanvas.FRAME18_NORM_CROP
            cx = int(img.width() * rx)
            cy = int(img.height() * ry)
            cw = int(img.width() * rw)
            ch = int(img.height() * rh)
            img = img.copy(QRect(cx, cy, cw, ch))
            self._current_crop_norm_rect = _MapCanvas.FRAME18_NORM_CROP
            return self._to_grayscale_pixmap(img), (cx, cy)
        except Exception as exc:
            logger.warning("Error loading frame pixmap '%s': %s", path, exc)
            return None, (0, 0)

    def _load_region_map_pixmap(self, nodes: List[Dict[str, Any]]) -> Tuple[QPixmap, Tuple[int, int]]:
        """
        Loads the region map image and converts it to grayscale based on _background_mode:
        - 'mosaic': Built from saved 80x80 node patches.
        - 'frame18': Cropped map region from video_frames/frame_18s.jpg.
        - 'hd': Altgard HD map from assets/maps/altgard.png (2258x2400) or region file.
        - 'auto': Priority: explicit -> HD -> reference frame -> mosaic.
        """
        mode = getattr(self, "_background_mode", "auto")
        safe_reg = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in (self._region_name or "Altgard"))
        # Görev 4: her yüklemede sıfırlanır; harita yoksa ilgili dal doldurur.
        self._map_missing_for_region = ""

        # 1. Mosaic mode
        if mode == "mosaic":
            self._current_crop_norm_rect = None
            return self._build_patch_mosaic_pixmap(nodes), (0, 0)

        # 2. Reference frame mode
        if mode == "frame18":
            # P3-6: referans kareler kök-bağımlı (eski: biri os.getcwd(),
            # biri CWD'ye göreli "video_frames/..." idi).
            for f_cand in (
                str(project_path("video_frames", "frame_18s.jpg")),
                str(project_path("video_frames", "frame_24s.jpg")),
            ):
                if os.path.exists(f_cand):
                    pix, off = self._load_frame_pixmap(f_cand)
                    if pix is not None and not pix.isNull():
                        return pix, off
            # Explicit mode requested but no reference frame exists: do NOT
            # silently fall through to auto mode.
            logger.warning(
                "Harita modu 'frame18' seçili ama referans kare dosyası bulunamadı; mozaik kullanılıyor."
            )
            self._current_crop_norm_rect = None
            return self._build_patch_mosaic_pixmap(nodes), (0, 0)

        # 3. HD Map mode
        if mode == "hd":
            # Görev 4: Altgard görseli yalnızca bölge Altgard (veya bilinmiyor)
            # ise adaydır; başka bölge için kendi haritası yoksa mozaik kullanılır.
            for hd_cand in (
                *region_map_candidates(self._region_name),
                str(project_path("routes", "maps", f"{safe_reg}.png")),
                str(project_path("routes", f"{safe_reg}_map.png")),
            ):
                if os.path.exists(hd_cand):
                    img = QImage(hd_cand)
                    if not img.isNull():
                        self._current_crop_norm_rect = None
                        return self._to_grayscale_pixmap(img), (0, 0)
            # Explicit mode requested but no HD map file exists: do NOT silently
            # fall through to auto mode (which would ignore the explicit choice).
            if not is_altgard_map_region(self._region_name):
                self._map_missing_for_region = str(self._region_name or "")
            logger.warning(
                "Harita modu 'hd' seçili ama '%s' için HD harita dosyası bulunamadı; mozaik kullanılıyor.",
                self._region_name,
            )
            self._current_crop_norm_rect = None
            return self._build_patch_mosaic_pixmap(nodes), (0, 0)

        # 4. Auto mode (fallback priority: explicit -> HD map -> reference frame -> mosaic)
        candidates: List[Tuple[str, bool]] = []
        if self._explicit_map_path and os.path.exists(self._explicit_map_path):
            is_f18 = "frame_18" in os.path.basename(self._explicit_map_path).lower()
            candidates.append((self._explicit_map_path, is_f18))

        # Check HD map paths for region (Görev 4: Altgard görseli yalnızca
        # bölge Altgard — veya bilinmiyor — ise adaydır; başka bölgede kendi
        # görseli yoksa mozaik yedeğe düşülür, yanlış harita gösterilmez).
        for rel_path in (
            *region_map_candidates(self._region_name),
            str(project_path("routes", "maps", f"{safe_reg}.png")),
            str(project_path("routes", "maps", f"{safe_reg}.jpg")),
            str(project_path("routes", f"{safe_reg}_map.png")),
        ):
            if os.path.exists(rel_path) and rel_path not in [c[0] for c in candidates]:
                candidates.append((rel_path, False))
        if not is_altgard_map_region(self._region_name) and not any(
            os.path.isfile(p) for p in region_map_candidates(self._region_name)
        ):
            self._map_missing_for_region = str(self._region_name or "")
            logger.warning(
                "'%s' bölgesi için HD harita görseli yok (assets/maps/%s.png); "
                "Altgard haritası kullanılmıyor, mozaik yedeğe düşülüyor.",
                self._region_name,
                str(self._region_name or "").strip().lower(),
            )

        # Reference frame fallback (frame_18s.jpg Altgard kırpmasıdır;
        # başka bölgede yedek olamaz).
        frame18_path = str(project_path("video_frames", "frame_18s.jpg"))
        if (
            os.path.exists(frame18_path)
            and frame18_path not in [c[0] for c in candidates]
            and is_altgard_map_region(self._region_name)
        ):
            candidates.append((frame18_path, True))

        self._current_crop_norm_rect = None
        for path, should_crop_f18 in candidates:
            try:
                img = QImage(path)
                if not img.isNull():
                    offset = (0, 0)
                    crop_norm = None
                    if should_crop_f18:
                        rx, ry, rw, rh = _MapCanvas.FRAME18_NORM_CROP
                        cx = int(img.width() * rx)
                        cy = int(img.height() * ry)
                        cw = int(img.width() * rw)
                        ch = int(img.height() * rh)
                        crop_rect = QRect(cx, cy, cw, ch)
                        img = img.copy(crop_rect)
                        offset = (cx, cy)
                        crop_norm = _MapCanvas.FRAME18_NORM_CROP
                    gray_pix = self._to_grayscale_pixmap(img)
                    if not gray_pix.isNull():
                        self._current_crop_norm_rect = crop_norm
                        return gray_pix, offset
            except Exception as exc:
                logger.warning("Failed loading candidate map image '%s': %s", path, exc)

        # Fallback to mosaic
        self._current_crop_norm_rect = None
        return self._build_patch_mosaic_pixmap(nodes), (0, 0)

    def _build_patch_mosaic_pixmap(self, nodes: List[Dict[str, Any]]) -> QPixmap:
        """
        Builds a grayscale map mosaic using saved 80x80 node patches in routes/patches/
        placed at each node's coordinates.

        P3-7 (v18): yerleştirme TEK formülle yapılır — 1180 eşiği kaldırıldı.
        Eskiden mx > 1180 ise ölçekli, altı ise ham piksel kullanılıyordu:
        mx=1180 -> 1180 ama mx=1181 -> ~1089 (91px geri sıçrama); mx=1280 ->
        1180 ama mx=1281 -> ~787 (393px geri sıçrama); my=720 -> 720 ama
        my=721 -> ~494. Şimdi norm_x/norm_y varsa doğrudan tuvale, yoksa
        dosyanın kanonik yardımcısı resolve_node_xy ile 1280x720 referans
        çerçeveye çekilip tek ölçekle tuvale taşınır — süreksiz eşik yok.
        """
        width, height = MOSAIC_CANVAS_WIDTH, MOSAIC_CANVAS_HEIGHT
        canvas_img = QImage(width, height, QImage.Format.Format_RGB32)
        canvas_img.fill(QColor("#262a30"))

        painter = QPainter(canvas_img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # Subtle topographic grid lines
        grid_pen = QPen(QColor("#2f343c"), 1)
        painter.setPen(grid_pen)
        for gx in range(0, width, 60):
            painter.drawLine(gx, 0, gx, height)
        for gy in range(0, height, 60):
            painter.drawLine(0, gy, width, gy)

        # P3-6: varsayılan yama dizini kök-bağımlı (eski: CWD'ye göreli
        # "routes/patches"). RouteManager.patch_dir göreliyse de köke göre
        # çözülür (RouteManager base_dir varsayılanı "routes" CWD'ye görelidir;
        # o varsayılan bu maddenin kapsamında değil, burada savunulur).
        patch_dir = str(project_path("routes", "patches"))
        if self.routes is not None and hasattr(self.routes, "patch_dir"):
            patch_dir = str(resolve_project_path(self.routes.patch_dir))

        for node in (nodes or []):
            patch_file = node.get("patch_file") or node.get("patch_path")
            if not patch_file:
                continue
            p_path = patch_file if os.path.isabs(patch_file) else os.path.join(patch_dir, os.path.basename(patch_file))
            if os.path.exists(p_path):
                try:
                    p_img = QImage(p_path)
                    if not p_img.isNull():
                        gray_patch = p_img.convertToFormat(QImage.Format.Format_Grayscale8)
                        nx_val = node.get("norm_x")
                        ny_val = node.get("norm_y")
                        if nx_val is not None and ny_val is not None and (0.0 < float(nx_val) <= 1.0 or 0.0 < float(ny_val) <= 1.0):
                            nx = int(round(float(nx_val) * width))
                            ny = int(round(float(ny_val) * height))
                        else:
                            rx, ry = resolve_node_xy(node, DEFAULT_MAP_REF[0], DEFAULT_MAP_REF[1])
                            if rx == 0.0 and ry == 0.0 and not any(
                                k in node for k in ("map_x", "map_y", "x", "y", "norm_x", "norm_y")
                            ):
                                # Hiç koordinatı olmayan düğüm eskisi gibi tuval ortasına konur.
                                rx, ry = DEFAULT_MAP_REF[0] / 2.0, DEFAULT_MAP_REF[1] / 2.0
                            nx = int(round(rx * width / DEFAULT_MAP_REF[0]))
                            ny = int(round(ry * height / DEFAULT_MAP_REF[1]))
                        pw, ph = gray_patch.width(), gray_patch.height()
                        painter.drawImage(nx - pw // 2, ny - ph // 2, gray_patch)
                except Exception as exc:
                    logger.debug("Failed to composite node patch '%s': %s", p_path, exc)

        painter.end()
        return QPixmap.fromImage(canvas_img)

    # -------------------------------------------------------------------------
    # AutoCartographer & Live Odometry Synchronization
    # -------------------------------------------------------------------------
    def set_auto_cartographer(self, cartographer: Any) -> None:
        """Attaches an AutoCartographer instance for real-time live auto-mapped point synchronization."""
        self.auto_cartographer = cartographer
        if hasattr(self, "_live_sync_timer") and self._live_sync_timer is not None and not self._live_sync_timer.isActive():
            self._live_sync_timer.start()
        self.reload_map_and_nodes()

    def start_live_sync(self, interval_ms: int = 1000) -> None:
        if hasattr(self, "_live_sync_timer") and self._live_sync_timer is not None:
            self._live_sync_timer.setInterval(interval_ms)
            if not self._live_sync_timer.isActive():
                self._live_sync_timer.start()

    def stop_live_sync(self) -> None:
        if hasattr(self, "_live_sync_timer") and self._live_sync_timer is not None:
            self._live_sync_timer.stop()

    def poll_live_updates(self) -> None:
        """Periodic heartbeat polling live auto-mapped nodes and player position from AutoCartographer / RouteManager."""
        try:
            cartographer = self.auto_cartographer
            if cartographer is None and self.routes is not None:
                cartographer = getattr(self.routes, "auto_cartographer", None) or getattr(self.routes, "_auto_cartographer", None)
                if cartographer is not None:
                    self.auto_cartographer = cartographer

            if cartographer is not None:
                mx = getattr(cartographer, "map_x", None)
                my = getattr(cartographer, "map_y", None)
                if mx is not None and my is not None:
                    # Live odometry carries no heading: keep the last known angle.
                    prev_angle = float(self._player_pos[2]) if self._player_pos is not None and len(self._player_pos) > 2 else 0.0
                    self.set_player_position(float(mx), float(my), prev_angle)

                reg = self.canonical_region_name(self._region_name)
                auto_nodes = []
                if hasattr(cartographer, "get_tracked_nodes"):
                    auto_nodes = cartographer.get_tracked_nodes(reg)
                elif hasattr(cartographer, "_tracked_nodes") and isinstance(cartographer._tracked_nodes, dict):
                    auto_nodes = cartographer._tracked_nodes.get(reg, [])

                curr_auto_count = len(auto_nodes)
                if curr_auto_count != self._last_auto_count:
                    self._last_auto_count = curr_auto_count
                    self.reload_map_and_nodes()
        except Exception as exc:
            logger.debug("Error in poll_live_updates: %s", exc)

    def closeEvent(self, event) -> None:
        try:
            if hasattr(self, "_live_sync_timer") and self._live_sync_timer is not None:
                self._live_sync_timer.stop()
        except Exception:
            pass
        super().closeEvent(event)

    # -------------------------------------------------------------------------
    # Canvas Signals & Interactive Node Callbacks
    # -------------------------------------------------------------------------
    def _connect_canvas_signals(self) -> None:
        """Dynamically connects all interaction signals from self.canvas to window handlers."""
        if self.canvas is None:
            return

        try:
            self.canvas.node_left_clicked.connect(self._on_node_left_clicked)
        except Exception:
            pass

        try:
            self.canvas.node_right_clicked.connect(self._on_node_right_clicked)
        except Exception:
            pass

        for sig_name in ("node_added", "node_created", "on_canvas_add_node"):
            sig = getattr(self.canvas, sig_name, None)
            if sig is not None and hasattr(sig, "connect"):
                try:
                    sig.connect(self._on_canvas_node_added)
                except Exception as exc:
                    logger.debug("Could not connect %s: %s", sig_name, exc)

        for sig_name in ("node_moved", "node_dragged", "node_position_changed"):
            sig = getattr(self.canvas, sig_name, None)
            if sig is not None and hasattr(sig, "connect"):
                try:
                    sig.connect(self._on_canvas_node_moved)
                except Exception as exc:
                    logger.debug("Could not connect %s: %s", sig_name, exc)

        for sig_name in ("node_deleted", "canvas_node_deleted"):
            sig = getattr(self.canvas, sig_name, None)
            if sig is not None and hasattr(sig, "connect"):
                try:
                    sig.connect(self._on_canvas_node_deleted)
                except Exception as exc:
                    logger.debug("Could not connect %s: %s", sig_name, exc)

        # NOTE: the canvas has no on_node_added / on_node_moved / on_node_deleted
        # callback attributes — those assignments were dead (nothing ever reads
        # them). Node add/move/delete arrive via the pyqtSignals connected above
        # and via the canvas' on_canvas_add_node hook; do not re-add dead hooks.

    def _on_canvas_node_added(self, data: Any, *args) -> None:
        """Slot for canvas node addition signal or callback."""
        try:
            if isinstance(data, (tuple, list)) and len(data) >= 2:
                self.add_node(float(data[0]), float(data[1]))
            elif isinstance(data, (int, float)) and args and len(args) >= 2:
                # Signature: (node_id, map_x, map_y)
                nid = int(data)
                nodes, _ = self._extract_region_nodes_and_route_ids()
                if not any(int(n.get("id", -1)) == nid for n in nodes):
                    self.add_node(float(args[0]), float(args[1]))
            elif isinstance(data, (int, float)) and args and len(args) == 1:
                # Signature: (map_x, map_y)
                self.add_node(float(data), float(args[0]))
            elif isinstance(data, dict):
                nid = data.get("id")
                nodes, _ = self._extract_region_nodes_and_route_ids()
                if nid is not None and any(int(n.get("id", -1)) == int(nid) for n in nodes):
                    return
                x = data.get("map_x", data.get("x", 0.0))
                y = data.get("map_y", data.get("y", 0.0))
                nx = data.get("norm_x")
                ny = data.get("norm_y")
                rtype = data.get("resource_type", "Orichalcum Ore")
                self.add_node(float(x), float(y), norm_x=nx, norm_y=ny, resource_type=rtype)
        except Exception as exc:
            logger.error("Error in _on_canvas_node_added: %s", exc)

    def _on_canvas_node_moved(self, node_id: int, new_x: float, new_y: float, *args) -> None:
        """Slot for canvas node drag-to-move signal or callback.

        Live drag update: refreshes the node position in memory only. The
        disk write happens exactly once on release (canvas
        _save_node_position -> move_node); saving here would write to disk
        on every mouse-move event during a drag.
        """
        try:
            norm_x = args[0] if len(args) > 0 else None
            norm_y = args[1] if len(args) > 1 else None
            reg = self.canonical_region_name(self._region_name)
            updated = False
            if self.routes is not None and hasattr(self.routes, "_regions") and isinstance(self.routes._regions, dict):
                reg_info = self.routes._regions.get(reg)
                if reg_info is not None:
                    for n in reg_info.get("nodes", []):
                        nid = getattr(n, "id", None) or (n.get("id") if isinstance(n, dict) else None)
                        if nid is not None and int(nid) == int(node_id):
                            if hasattr(n, "map_x"):
                                n.map_x = int(round(float(new_x)))
                                n.map_y = int(round(float(new_y)))
                                if norm_x is not None:
                                    n.norm_x = round(float(norm_x), 4)
                                if norm_y is not None:
                                    n.norm_y = round(float(norm_y), 4)
                            if isinstance(n, dict):
                                n["map_x"] = int(round(float(new_x)))
                                n["map_y"] = int(round(float(new_y)))
                                if norm_x is not None:
                                    n["norm_x"] = round(float(norm_x), 4)
                                if norm_y is not None:
                                    n["norm_y"] = round(float(norm_y), 4)
                            updated = True
                            break
            if self.canvas is not None and hasattr(self.canvas, "_nodes"):
                for cn in self.canvas._nodes:
                    if int(cn.get("id", -1)) == int(node_id):
                        cn["map_x"] = int(round(float(new_x)))
                        cn["map_y"] = int(round(float(new_y)))
                        if norm_x is not None:
                            cn["norm_x"] = round(float(norm_x), 4)
                        if norm_y is not None:
                            cn["norm_y"] = round(float(norm_y), 4)
                        updated = True
                        break
                self.canvas.update()
            if not updated:
                logger.debug("Dragged node #%d not found in region '%s'; no update.", node_id, reg)
        except Exception as exc:
            logger.error("Error in _on_canvas_node_moved: %s", exc)

    def _on_canvas_node_deleted(self, node_id: int) -> None:
        """Slot for canvas node deletion signal or callback."""
        try:
            # Guard: canvas.delete_node() already delegates to the window, so
            # this slot must not delete the same node a second time.
            nodes, _ = self._extract_region_nodes_and_route_ids()
            if not any(int(n.get("id", -1)) == int(node_id) for n in nodes):
                return
            self.delete_node(int(node_id))
        except Exception as exc:
            logger.error("Error in _on_canvas_node_deleted: %s", exc)

    def add_node(
        self,
        map_x: float,
        map_y: float,
        norm_x: Optional[float] = None,
        norm_y: Optional[float] = None,
        resource_type: str = "Orichalcum Ore",
        routes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Adds a new node to the active region, atomically updates self.routes._regions[reg]['nodes'],
        and immediately saves to disk via self.routes.save_region(reg).
        """
        reg = self.canonical_region_name(self._region_name)

        pw = float(self.canvas._base_pixmap.width()) if (self.canvas and self.canvas._base_pixmap and not self.canvas._base_pixmap.isNull()) else 1920.0
        ph = float(self.canvas._base_pixmap.height()) if (self.canvas and self.canvas._base_pixmap and not self.canvas._base_pixmap.isNull()) else 1080.0

        if 0.0 <= float(map_x) <= 1.0 and 0.0 <= float(map_y) <= 1.0 and norm_x is None and norm_y is None:
            norm_x = float(map_x)
            norm_y = float(map_y)
            map_x = norm_x * pw
            map_y = norm_y * ph
        else:
            if norm_x is None:
                norm_x = round(float(map_x) / pw, 4)
            if norm_y is None:
                norm_y = round(float(map_y) / ph, 4)

        default_route = "All"
        if self.routes is not None and hasattr(self.routes, "DEFAULT_ROUTE"):
            default_route = str(self.routes.DEFAULT_ROUTE)
        target_routes = routes or ([self._route_name] if self._route_name and self._route_name != "All" else [default_route])

        existing_nodes = []
        if self.routes is not None:
            if hasattr(self.routes, "_regions") and isinstance(self.routes._regions, dict):
                reg_info = self.routes._regions.setdefault(reg, {"routes": ["All", default_route], "nodes": []})
                existing_nodes = reg_info.setdefault("nodes", [])
            elif hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                reg_data = self.routes.data.setdefault("regions", {}).setdefault(reg, {"routes": {}, "nodes": []})
                existing_nodes = reg_data.setdefault("nodes", [])

        existing_ids = [
            int(getattr(n, "id", 0) if hasattr(n, "id") else (n.get("id", 0) if isinstance(n, dict) else 0))
            for n in existing_nodes
        ]
        new_id = (max(existing_ids) + 1) if existing_ids else 1

        node_dict = {
            "id": new_id,
            "region": reg,
            "routes": list(target_routes),
            # Repo-göreli saklanır (kalıcı JSON makineden bağımsız kalsın);
            # okuyan taraf resolve_node_patch_path ile köke göre çözer (P3-6).
            "patch_path": f"routes/{reg}/patches/node_{new_id}.png",
            "in_air": False,
            "map_x": int(round(float(map_x))),
            "map_y": int(round(float(map_y))),
            "norm_x": round(float(norm_x), 4),
            "norm_y": round(float(norm_y), 4),
            "resource_type": str(resource_type or "Orichalcum Ore"),
            "max_harvests": 3,
            "current_harvests": 0,
            "source_type": "manual",
            "state": "active",
            "seq_index": len(existing_nodes) + 1,
            "last_visited_timestamp": 0.0,
            "rest_duration_sec": self._rest_duration_sec,
            "not_gatherable": False,
        }

        # Determine whether region nodes are stored as NodeData or dicts
        use_node_data = False
        if NodeData is not None:
            if existing_nodes and isinstance(existing_nodes[0], NodeData):
                use_node_data = True
            elif not existing_nodes and (hasattr(self.routes, "add_node") or hasattr(self.routes, "base_dir")):
                use_node_data = True

        new_obj = node_dict
        if use_node_data and NodeData is not None:
            try:
                new_obj = NodeData.from_dict(node_dict)
            except Exception as exc:
                logger.debug("NodeData.from_dict fallback to dict: %s", exc)
                new_obj = node_dict

        if self.routes is not None:
            if hasattr(self.routes, "_regions") and isinstance(self.routes._regions, dict):
                self.routes._regions[reg]["nodes"].append(new_obj)
                if hasattr(self.routes, "save_region"):
                    self.routes.save_region(reg)
                elif hasattr(self.routes, "save_data"):
                    self.routes.save_data()
            elif hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                reg_data = self.routes.data.setdefault("regions", {}).setdefault(reg, {})
                reg_nodes = reg_data.setdefault("nodes", [])
                reg_nodes.append(node_dict)
                if hasattr(self.routes, "save_data"):
                    self.routes.save_data()

        logger.info("Nokta #%d (%s, %s) bölgeye eklendi: '%s' ve diske kaydedildi.", new_id, map_x, map_y, reg)
        self.reload_map_and_nodes()
        self.node_added.emit(node_dict)
        return node_dict

    def move_node(
        self,
        node_id: int,
        new_x: float,
        new_y: float,
        norm_x: Optional[float] = None,
        norm_y: Optional[float] = None,
    ) -> bool:
        """
        Updates the position of a node, atomically updates self.routes._regions[reg]['nodes'],
        and immediately saves to disk via self.routes.save_region(reg).
        """
        reg = self.canonical_region_name(self._region_name)
        pw = float(self.canvas._base_pixmap.width()) if (self.canvas and self.canvas._base_pixmap and not self.canvas._base_pixmap.isNull()) else 1920.0
        ph = float(self.canvas._base_pixmap.height()) if (self.canvas and self.canvas._base_pixmap and not self.canvas._base_pixmap.isNull()) else 1080.0

        if 0.0 <= float(new_x) <= 1.0 and 0.0 <= float(new_y) <= 1.0 and norm_x is None and norm_y is None:
            norm_x = float(new_x)
            norm_y = float(new_y)
            new_x = norm_x * pw
            new_y = norm_y * ph
        else:
            if norm_x is None:
                norm_x = round(float(new_x) / pw, 4)
            if norm_y is None:
                norm_y = round(float(new_y) / ph, 4)

        found = False
        saved = False
        if self.routes is not None:
            if hasattr(self.routes, "_regions") and isinstance(self.routes._regions, dict):
                reg_info = self.routes._regions.get(reg)
                if reg_info is not None:
                    for n in reg_info.get("nodes", []):
                        nid = getattr(n, "id", None) or (n.get("id") if isinstance(n, dict) else None)
                        if nid is not None and int(nid) == int(node_id):
                            if hasattr(n, "map_x"):
                                n.map_x = int(round(float(new_x)))
                                n.map_y = int(round(float(new_y)))
                                n.norm_x = round(float(norm_x), 4)
                                n.norm_y = round(float(norm_y), 4)
                            if isinstance(n, dict):
                                n["map_x"] = int(round(float(new_x)))
                                n["map_y"] = int(round(float(new_y)))
                                n["norm_x"] = round(float(norm_x), 4)
                                n["norm_y"] = round(float(norm_y), 4)
                            found = True
                            break

                if hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                    legacy_nodes = self.routes.data.get("regions", {}).get(reg, {}).get("nodes", [])
                    for n in legacy_nodes:
                        nid = getattr(n, "id", None) or (n.get("id") if isinstance(n, dict) else None)
                        if nid is not None and int(nid) == int(node_id):
                            if isinstance(n, dict):
                                n["map_x"] = int(round(float(new_x)))
                                n["map_y"] = int(round(float(new_y)))
                                n["norm_x"] = round(float(norm_x), 4)
                                n["norm_y"] = round(float(norm_y), 4)
                            elif hasattr(n, "map_x"):
                                n.map_x = int(round(float(new_x)))
                                n.map_y = int(round(float(new_y)))
                                n.norm_x = round(float(norm_x), 4)
                                n.norm_y = round(float(norm_y), 4)

                # Only persist when the node was actually found: saving (and
                # logging "saved to disk") for a missing node is misleading.
                if found:
                    if hasattr(self.routes, "save_region"):
                        # K-19: yalnızca açık False başarısızlıktır (None
                        # dönen eski/mock kayıtçılar başarılı sayılır);
                        # False ise "kaydedildi" logu ve node_moved yok.
                        saved = self.routes.save_region(reg) is not False
                    elif hasattr(self.routes, "save_data"):
                        saved = self.routes.save_data() is not False

            elif hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                reg_data = self.routes.data.get("regions", {}).get(reg, {})
                for n in reg_data.get("nodes", []):
                    if int(n.get("id", -1)) == int(node_id):
                        n["map_x"] = int(round(float(new_x)))
                        n["map_y"] = int(round(float(new_y)))
                        n["norm_x"] = round(float(norm_x), 4)
                        n["norm_y"] = round(float(norm_y), 4)
                        found = True
                        break
                if found and hasattr(self.routes, "save_data"):
                    saved = self.routes.save_data() is not False

        # No RouteManager attached: the canvas is the only store — a node that
        # exists on the canvas can still be moved in memory (no disk save).
        if not found and self.routes is None and self.canvas is not None and hasattr(self.canvas, "_nodes"):
            if any(int(cn.get("id", -1)) == int(node_id) for cn in self.canvas._nodes if isinstance(cn, dict)):
                found = True

        if not found:
            logger.warning(
                "Taşınacak düğüm bulunamadı: #%d (%s); konum güncellenmedi, diske yazılmadı.",
                node_id, reg,
            )
            return False

        if self.canvas is not None and hasattr(self.canvas, "_nodes"):
            for cn in self.canvas._nodes:
                if int(cn.get("id", -1)) == int(node_id):
                    cn["map_x"] = int(round(float(new_x)))
                    cn["map_y"] = int(round(float(new_y)))
                    cn["norm_x"] = round(float(norm_x), 4)
                    cn["norm_y"] = round(float(norm_y), 4)
                    break
            self.canvas.update()

        if saved:
            logger.info("Nokta #%d (%s, %s) koordinatlarına taşındı (%s) ve diske kaydedildi.", node_id, new_x, new_y, reg)
        elif self.routes is not None:
            # K-19: yazım başarısızken "taşındı/kaydedildi" denmez ve
            # node_moved yayılmaz — dinleyenler eski konumu korur.
            logger.error(
                "Nokta #%d (%s, %s) diske yazılamadı (%s); konum yalnızca bellekte güncellendi, node_moved yayılmadı.",
                node_id, new_x, new_y, reg,
            )
            return False
        else:
            logger.info("Nokta #%d (%s, %s) koordinatlarına taşındı (%s); kalıcı kayıt yok (RouteManager bağlı değil).", node_id, new_x, new_y, reg)
        self.node_moved.emit(int(node_id), float(new_x), float(new_y))
        return found

    def get_node_patch_path(self, node_id: int) -> str:
        """Returns the resolved disk path of routes/<region>/patches/node_<id>.png."""
        nodes, _ = self._extract_region_nodes_and_route_ids()
        node = next((n for n in nodes if int(n.get("id", 0)) == int(node_id)), {"id": node_id})
        if self.canvas is not None and hasattr(self.canvas, "resolve_node_patch_path"):
            return self.canvas.resolve_node_patch_path(node, self._region_name)
        return str(project_path("routes", self._region_name or "Altgard", "patches", f"node_{node_id}.png"))

    def get_node_patch_pixmap(self, node_id: int, size: int = 72) -> QPixmap:
        """Loads and returns the 80x80 node patch preview thumbnail."""
        nodes, _ = self._extract_region_nodes_and_route_ids()
        node = next((n for n in nodes if int(n.get("id", 0)) == int(node_id)), {"id": node_id})
        if self.canvas is not None and hasattr(self.canvas, "get_node_patch_pixmap"):
            return self.canvas.get_node_patch_pixmap(node, size=size)
        return QPixmap()

    # -------------------------------------------------------------------------
    # Data Synchronization & Node Operations
    # -------------------------------------------------------------------------
    def reload_map_and_nodes(self) -> None:
        """Reloads region nodes, active route membership, live auto-mapped points, and grayscale map."""
        try:
            self._region_name = self.canonical_region_name(self._region_name or self._resolve_active_region())
            self._route_name = self._route_name or self._resolve_active_route()
            self.setWindowTitle(self._build_title_text())
            if hasattr(self, "lbl_region_value"):
                self.lbl_region_value.setText(str(self._region_name))
            if hasattr(self, "lbl_route_value"):
                self.lbl_route_value.setText(str(self._route_name))

            nodes, route_ids = self._extract_region_nodes_and_route_ids()

            # Merge live AutoCartographer tracked points
            cartographer = self.auto_cartographer
            if cartographer is None and self.routes is not None:
                cartographer = getattr(self.routes, "auto_cartographer", None) or getattr(self.routes, "_auto_cartographer", None)
                if cartographer is not None:
                    self.auto_cartographer = cartographer

            if cartographer is not None:
                reg = self.canonical_region_name(self._region_name)
                tracked_raw = []
                if hasattr(cartographer, "get_tracked_nodes"):
                    tracked_raw = cartographer.get_tracked_nodes(reg)
                elif hasattr(cartographer, "_tracked_nodes") and isinstance(cartographer._tracked_nodes, dict):
                    tracked_raw = cartographer._tracked_nodes.get(reg, [])

                existing_ids = {int(n.get("id", 0)) for n in nodes}
                for t in tracked_raw:
                    t_dict = t.to_dict() if hasattr(t, "to_dict") else dict(t)
                    tid = int(t_dict.get("id", t_dict.get("node_id", 0)))
                    t_dict["source_type"] = "minimap_auto"
                    t_dict.setdefault("harvest_badge", "0/3")
                    if tid not in existing_ids:
                        existing_ids.add(tid)
                        nodes.append(t_dict)
                        route_ids.append(tid)

                mx = getattr(cartographer, "map_x", None)
                my = getattr(cartographer, "map_y", None)
                if mx is not None and my is not None:
                    # Live odometry carries no heading: keep the last known angle.
                    prev_angle = float(self._player_pos[2]) if self._player_pos is not None and len(self._player_pos) > 2 else 0.0
                    self._player_pos = (float(mx), float(my), prev_angle)

            # Update auto-mapped badge count in UI
            total_auto = sum(1 for n in nodes if n.get("source_type") == "minimap_auto" or n.get("is_auto"))
            if hasattr(self, "lbl_badge_minimap_auto"):
                badge_text = f"◎ Minimap/3B Oto-Kayıt ({total_auto})" if total_auto > 0 else "◎ Minimap/3B Oto-Kayıt"
                self.lbl_badge_minimap_auto.setText(badge_text)

            pixmap, offset = self._load_region_map_pixmap(nodes)
            crop_norm = getattr(self, "_current_crop_norm_rect", None)

            # Check if node coordinates are already relative to the cropped map or full-screen
            if offset != (0, 0) and nodes:
                ox, oy = offset
                has_fullscreen = any(
                    float(n.get("map_x", n.get("x", 0))) >= ox or float(n.get("map_y", n.get("y", 0))) >= oy
                    for n in nodes
                )
                if not has_fullscreen:
                    offset = (0, 0)
                    crop_norm = None

            self.canvas.set_map_pixmap(pixmap, coord_offset=offset, crop_norm_rect=crop_norm)
            self.canvas.set_nodes_data(nodes, route_ids, rest_duration_sec=self._rest_duration_sec)
            if self._player_pos is not None:
                angle = float(self._player_pos[2]) if len(self._player_pos) > 2 else 0.0
                self.canvas.set_player_position(self._player_pos[0], self._player_pos[1], angle)
            self._populate_material_filter_combo()
        except Exception as exc:
            logger.error("Error in MapViewerWindow.reload_map_and_nodes: %s", exc)

    def _extract_region_nodes_and_route_ids(self) -> Tuple[List[Dict[str, Any]], List[int]]:
        nodes: List[Dict[str, Any]] = []
        route_ids: List[int] = []

        if self.routes is None:
            return nodes, route_ids

        try:
            target_reg = self.canonical_region_name(self._region_name or self._resolve_active_region())
            self._region_name = target_reg

            # 1. Authoritative _regions check (primary)
            if hasattr(self.routes, "_regions") and isinstance(self.routes._regions, dict) and self.routes._regions:
                regions = self.routes._regions
                reg_info = regions.get(target_reg)
                if reg_info is None:
                    # Case-insensitive / canonical lookup
                    for rk, rv in regions.items():
                        if rk.lower() == target_reg.lower() or self.canonical_region_name(rk).lower() == target_reg.lower():
                            reg_info = rv
                            target_reg = rk
                            self._region_name = target_reg
                            break

                if reg_info is None and regions:
                    # Fallback to first available region only if canonical target not present
                    first_reg = next(iter(regions.keys()))
                    logger.warning(
                        "Bölge '%s' bulunamadı; ilk bölge '%s' kullanılıyor.",
                        target_reg, first_reg,
                    )
                    self._region_name = first_reg
                    target_reg = first_reg
                    self.setWindowTitle(self._build_title_text())
                    if hasattr(self, "lbl_region_value"):
                        self.lbl_region_value.setText(str(self._region_name))
                    reg_info = regions.get(first_reg, {})

                if isinstance(reg_info, dict):
                    raw_nodes = reg_info.get("nodes", [])
                    for item in raw_nodes:
                        if isinstance(item, dict):
                            nodes.append(dict(item))
                        elif hasattr(item, "to_dict") and callable(item.to_dict):
                            nodes.append(dict(item.to_dict()))

                    if self._route_name and self._route_name != "All":
                        for n in raw_nodes:
                            r_list = getattr(n, "routes", None) or (n.get("routes") if isinstance(n, dict) else [])
                            nid = getattr(n, "id", None) or (n.get("id") if isinstance(n, dict) else None)
                            if nid is not None and self._route_name in (r_list or []):
                                route_ids.append(int(nid))
                    else:
                        # 'All' is the virtual route: every node whose in_route
                        # flag is not False belongs to it.
                        for n in raw_nodes:
                            nid = getattr(n, "id", None) or (n.get("id") if isinstance(n, dict) else None)
                            in_route = getattr(n, "in_route", None) if hasattr(n, "in_route") else (n.get("in_route", True) if isinstance(n, dict) else True)
                            if nid is not None and in_route is not False:
                                route_ids.append(int(nid))
                    return nodes, route_ids

            # 2. Legacy fallback to self.routes.data
            if hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                regions = self.routes.data.get("regions", {})
                reg_data = regions.get(target_reg)
                if reg_data is None:
                    for rk, rv in regions.items():
                        if rk.lower() == target_reg.lower() or self.canonical_region_name(rk).lower() == target_reg.lower():
                            reg_data = rv
                            target_reg = rk
                            self._region_name = target_reg
                            break

                if reg_data is None and regions:
                    first_reg = next(iter(regions.keys()))
                    logger.warning(
                        "Bölge '%s' bulunamadı; ilk bölge '%s' kullanılıyor.",
                        target_reg, first_reg,
                    )
                    self._region_name = first_reg
                    target_reg = first_reg
                    self.setWindowTitle(self._build_title_text())
                    if hasattr(self, "lbl_region_value"):
                        self.lbl_region_value.setText(str(self._region_name))
                    reg_data = regions.get(first_reg, {})

                if isinstance(reg_data, dict):
                    raw_nodes = reg_data.get("nodes", [])
                    if isinstance(raw_nodes, list):
                        for item in raw_nodes:
                            if isinstance(item, dict):
                                nodes.append(item)
                            elif hasattr(item, "to_dict") and callable(item.to_dict):
                                nodes.append(dict(item.to_dict()))
                    routes_dict = reg_data.get("routes", {})
                    if isinstance(routes_dict, dict):
                        if self._route_name in routes_dict:
                            route_ids = [int(x) for x in routes_dict.get(self._route_name, [])]
                        elif "All" in routes_dict:
                            route_ids = [int(x) for x in routes_dict.get("All", [])]
                        else:
                            route_ids = [int(n.get("id", 0)) for n in nodes]
            elif hasattr(self.routes, "get_route_nodes"):
                raw_list = list(self.routes.get_route_nodes() or [])
                for item in raw_list:
                    if isinstance(item, dict):
                        nodes.append(item)
                    elif hasattr(item, "to_dict") and callable(item.to_dict):
                        nodes.append(dict(item.to_dict()))
                route_ids = [int(n.get("id", 0)) for n in nodes]
        except Exception as exc:
            logger.error("Failed extracting nodes from RouteManager: %s", exc)

        return nodes, route_ids

    def optimize_sequential_route(self) -> List[Dict[str, Any]]:
        """
        'Sıralı Rota Optimize Et' butonu aksiyonu:
        Harita üzerindeki noktaları zikzaksız en yakın komşu + 2-Opt sırasına dizer.
        Authoritative RouteManager._regions[self._region_name]["nodes"] yapısını
        günceller (seq_index 1..N) ve save_region(self._region_name) çağırarak diske kaydeder.
        """
        try:
            # Only nodes that belong to the active route are optimized. The
            # canvas holds ALL region nodes (including ones removed from the
            # route), so optimizing the full canvas list would silently put
            # out-of-route nodes back into the route.
            canvas_nodes = list(self.canvas._nodes)
            route_ids = list(getattr(self.canvas, "_route_node_ids_list", []) or [])
            if route_ids:
                route_set = set(route_ids)
                nodes = [n for n in canvas_nodes if int(n.get("id", 0) if isinstance(n, dict) else getattr(n, "id", 0)) in route_set]
            else:
                nodes = [
                    n for n in canvas_nodes
                    if (n.get("in_route", True) if isinstance(n, dict) else getattr(n, "in_route", True)) is not False
                ]
            if not nodes:
                # F-0532: empty route — route_ids is empty and every region
                # node has in_route False. There is nothing to optimise, and
                # falling back to the full canvas list would write the
                # removed nodes back into the route (routes_dict + the
                # set_nodes_data call at the end of this function). No-op
                # instead: no persistence, no canvas mutation.
                logger.info(
                    "Sıralı rota optimize edilmedi: rota boş (%s).", self._route_name
                )
                return []
            # P1-7 (K-20): norm/piksel birimleri ayrılır, çözünürlük
            # kanvastan okunur. Kurallar:
            #  * norm_x VE norm_y ikisi de [0,1] içindeyse norm dalı —
            #    kanvas boyutuyla ölçeklenir (eski `or` koşulu karışık
            #    birimli düğümü de norm sayıp yanlış ölçekliyordu; eski
            #    sabit 1920x1080 kaldırıldı, kanvas yoksa 1280x720).
            #  * değilse piksel dalı (map_x/map_y); ikisi de 0 ise düğüm
            #    koordinatsızdır -> None (turdan çıkarılır, (0,0)
            #    hayaleti olarak sıralamaya girmez).
            _canvas_w, _canvas_h = DEFAULT_MAP_REF
            try:
                _pix = self.canvas._base_pixmap if self.canvas is not None else None
                if _pix is not None and not _pix.isNull():
                    _canvas_w = float(_pix.width())
                    _canvas_h = float(_pix.height())
            except Exception:
                pass

            def _node_dist_xy(n: Dict[str, Any]) -> Optional[Tuple[float, float]]:
                nx_val = n.get("norm_x")
                ny_val = n.get("norm_y")
                if nx_val is not None and ny_val is not None:
                    try:
                        nx_f = float(nx_val)
                        ny_f = float(ny_val)
                        if 0.0 <= nx_f <= 1.0 and 0.0 <= ny_f <= 1.0:
                            # (0.5, 0.5) yer tutucudur — norm dalı reddedilir
                            # ama piksel koordinat varsa piksel dalına düşülür.
                            if not (nx_f == 0.5 and ny_f == 0.5):
                                return nx_f * _canvas_w, ny_f * _canvas_h
                    except (ValueError, TypeError):
                        pass
                mx = float(n.get("map_x", n.get("x", 0.0)) or 0.0)
                my = float(n.get("map_y", n.get("y", 0.0)) or 0.0)
                if mx == 0.0 and my == 0.0:
                    return None
                return mx, my

            _coordless = [n for n in nodes if _node_dist_xy(n) is None]
            if _coordless:
                logger.warning(
                    "optimize_sequential_route: %d koordinatsız düğüm "
                    "sıralamadan çıkarıldı (id=%s).",
                    len(_coordless),
                    [n.get("id") for n in _coordless],
                )
                nodes = [n for n in nodes if _node_dist_xy(n) is not None]

            if len(nodes) <= 2:
                return nodes

            # 1. Nearest Neighbor Ordering
            unvisited = list(nodes)
            unvisited.sort(
                key=lambda n: _node_dist_xy(n)[0] * 0.7 + _node_dist_xy(n)[1] * 0.3  # type: ignore[index]
            )
            ordered = [unvisited.pop(0)]
            while unvisited:
                curr = ordered[-1]
                cx, cy = _node_dist_xy(curr)
                nxt = min(
                    unvisited,
                    key=lambda cand: math.hypot(
                        _node_dist_xy(cand)[0] - cx,
                        _node_dist_xy(cand)[1] - cy,
                    ),
                )
                ordered.append(nxt)
                unvisited.remove(nxt)

            # 2. 2-Opt Pass to uncross overlapping segments
            if len(ordered) >= 4:
                coords = [_node_dist_xy(n) for n in ordered]
                n_count = len(ordered)
                max_passes = 40
                passes = 0
                improved = True
                while improved and passes < max_passes:
                    improved = False
                    passes += 1
                    for i in range(1, n_count - 1):
                        for j in range(i + 1, n_count):
                            if j - i == 1:
                                continue
                            d_before = (
                                math.hypot(coords[i - 1][0] - coords[i][0], coords[i - 1][1] - coords[i][1])
                                + math.hypot(coords[j - 1][0] - coords[j % n_count][0], coords[j - 1][1] - coords[j % n_count][1])
                            )
                            d_after = (
                                math.hypot(coords[i - 1][0] - coords[j - 1][0], coords[i - 1][1] - coords[j - 1][1])
                                + math.hypot(coords[i][0] - coords[j % n_count][0], coords[i][1] - coords[j % n_count][1])
                            )
                            if d_after < d_before - 1e-4:
                                ordered[i:j] = list(reversed(ordered[i:j]))
                                coords[i:j] = list(reversed(coords[i:j]))
                                improved = True
                                break
                        if improved:
                            break

            # 3. Assign sequential seq_index (1..N) to ordered list
            for idx, n in enumerate(ordered, start=1):
                if isinstance(n, dict):
                    n["seq_index"] = idx
                elif hasattr(n, "seq_index"):
                    n.seq_index = idx

            # 4. Authoritative RouteManager._regions update & save_region
            if self.routes is not None:
                if hasattr(self.routes, "_regions") and isinstance(self.routes._regions, dict) and self._region_name in self.routes._regions:
                    reg_entry = self.routes._regions[self._region_name]
                    existing_nodes = reg_entry.get("nodes", [])

                    # Map existing NodeData by ID
                    existing_node_map = {}
                    for nd in existing_nodes:
                        nid = getattr(nd, "id", None) or (nd.get("id") if isinstance(nd, dict) else None)
                        if nid is not None:
                            existing_node_map[int(nid)] = nd

                    ordered_node_objs = []
                    ordered_ids = set()
                    for idx, n in enumerate(ordered, start=1):
                        nid = int(n.get("id", 0) if isinstance(n, dict) else getattr(n, "id", 0))
                        node_obj = existing_node_map.get(nid)
                        if node_obj is not None:
                            if hasattr(node_obj, "seq_index"):
                                node_obj.seq_index = idx
                            if isinstance(node_obj, dict):
                                node_obj["seq_index"] = idx
                            ordered_ids.add(nid)
                            ordered_node_objs.append(node_obj)
                        elif isinstance(n, dict) and _MapCanvas._is_minimap_auto_node(n):
                            # S044 (v19/P3-8): eski hali koşullu ifadeydi
                            # (`... if isinstance(n, dict) else False`) —
                            # okunması ve önceliği belirsizdi. Eşdeğer açık
                            # yazım: dict değilse False, dict ise kontrol.
                            # Temporary live AutoCartographer points (source_type
                            # 'minimap_auto') are display-only: never persist
                            # them as permanent region nodes.
                            continue
                        else:
                            if NodeData is not None and isinstance(n, dict):
                                new_nd = NodeData.from_dict(n)
                                new_nd.seq_index = idx
                                ordered_ids.add(nid)
                                ordered_node_objs.append(new_nd)
                            else:
                                if isinstance(n, dict):
                                    n["seq_index"] = idx
                                ordered_ids.add(nid)
                                ordered_node_objs.append(n)

                    # Append any region nodes not in ordered
                    next_idx = len(ordered_node_objs) + 1
                    for nd in existing_nodes:
                        nid = getattr(nd, "id", None) or (nd.get("id") if isinstance(nd, dict) else None)
                        if nid is not None and int(nid) not in ordered_ids:
                            if hasattr(nd, "seq_index"):
                                nd.seq_index = next_idx
                            elif isinstance(nd, dict):
                                nd["seq_index"] = next_idx
                            next_idx += 1
                            ordered_node_objs.append(nd)

                    # Authoritative assignment
                    reg_entry["nodes"] = ordered_node_objs

                    # Also update legacy data structure for callers reading self.routes.data
                    if hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                        reg_data = self.routes.data.get("regions", {}).get(self._region_name, {})
                        if isinstance(reg_data, dict):
                            reg_data["nodes"] = [
                                n.to_dict() if hasattr(n, "to_dict") else n for n in ordered_node_objs
                            ]
                            routes_dict = reg_data.setdefault("routes", {})
                            # Only persistent region nodes go into the saved
                            # route list (no temporary minimap_auto points),
                            # keeping the optimized sequence order.
                            routes_dict[self._route_name] = [
                                int(n.get("id", 0) if isinstance(n, dict) else getattr(n, "id", 0))
                                for n in ordered
                                if int(n.get("id", 0) if isinstance(n, dict) else getattr(n, "id", 0)) in ordered_ids
                            ]

                    # Save authoritative region to disk
                    if hasattr(self.routes, "save_region"):
                        self.routes.save_region(self._region_name)
                    elif hasattr(self.routes, "save_data"):
                        self.routes.save_data()

                elif hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                    # Legacy fallback (temporary minimap_auto points are
                    # display-only and must not be persisted).
                    persistent_ordered = [
                        n for n in ordered
                        if not (isinstance(n, dict) and _MapCanvas._is_minimap_auto_node(n))
                    ]
                    reg_data = self.routes.data.get("regions", {}).get(self._region_name, {})
                    if isinstance(reg_data, dict):
                        reg_data["nodes"] = list(persistent_ordered)
                        routes_dict = reg_data.setdefault("routes", {})
                        routes_dict[self._route_name] = [int(n.get("id", 0)) for n in persistent_ordered]
                        if hasattr(self.routes, "save_data"):
                            self.routes.save_data()

            route_ids = [int(n.get("id", 0) if isinstance(n, dict) else getattr(n, "id", 0)) for n in ordered]
            # F-0532: the canvas holds ALL region nodes — only route_ids
            # marks route membership. Passing just `ordered` would drop
            # out-of-route nodes from the canvas until the next reload, so
            # keep them (in their original relative order) after the
            # optimised route nodes.
            ordered_id_set = set(route_ids)
            remaining = [
                n for n in canvas_nodes
                if int(n.get("id", 0) if isinstance(n, dict) else getattr(n, "id", 0)) not in ordered_id_set
            ]
            self.canvas.set_nodes_data(
                list(ordered) + remaining, route_ids, rest_duration_sec=self._rest_duration_sec
            )
            logger.info("Sıralı rota optimize edildi (%d nokta).", len(ordered))
            return ordered
        except Exception as exc:
            logger.error("optimize_sequential_route hatası: %s", exc)
            return list(self.canvas._nodes)

    def set_material_filter(self, material: Union[str, Sequence[str], None]) -> None:
        """Sets the material filter in the combo box and updates canvas."""
        if material is None:
            text = "Tümü"
        elif isinstance(material, (list, tuple, set)):
            # Keep every selected material: taking only the first silently
            # dropped the rest of the selection.
            clean = [str(m).strip() for m in material if str(m).strip()]
            text = ", ".join(clean) if clean else "Tümü"
        else:
            text = str(material).strip()

        # Update combo text if present
        self._selected_material_filter = text
        combo = self.__dict__.get("combo_material_filter")
        if combo is not None:
            idx = -1
            norm_target = normalize_material_token(text)
            for i in range(combo.count()):
                item_text = combo.itemText(i)
                item_data = combo.itemData(i)
                if (item_data and normalize_material_token(str(item_data)) == norm_target) or \
                   item_text == text or normalize_material_token(item_text) == norm_target:
                    idx = i
                    break
            if idx >= 0:
                combo.setCurrentIndex(idx)
            else:
                combo.addItem(text, userData=text)
                combo.setCurrentIndex(combo.count() - 1)
        canvas = self.__dict__.get("canvas")
        if canvas is not None:
            canvas.set_material_filter(text)

    def get_material_filter(self) -> str:
        """Returns the current material filter string."""
        combo = self.__dict__.get("combo_material_filter")
        if combo is not None:
            data = combo.currentData()
            if data is not None and str(data).strip():
                return str(data)
            text = combo.currentText()
            clean = re.sub(r"^[^\w\s/]+", "", text).strip()
            return clean or text
        canvas = self.__dict__.get("canvas")
        if canvas is not None:
            return str(canvas.get_material_filter() or "Tümü")
        return str(self.__dict__.get("_selected_material_filter") or "Tümü")

    def draw_proximity_route(self) -> List[Dict[str, Any]]:
        """Calculates and renders proximity route for currently selected material."""
        mat = self.get_material_filter()
        canvas = self.__dict__.get("canvas")
        if canvas is None:
            return []
        canvas.set_material_filter(mat)
        ordered = canvas.calculate_proximity_route()
        canvas.update()
        logger.info("Seç-Topla yakınlık rotası oluşturuldu: %s (%d nokta)", mat, len(ordered))
        return ordered

    def filter_nodes_by_selected_material(self, material_query: str = "") -> List[Dict[str, Any]]:
        """
        Filters active region nodes by the given material query ('Orichalcum Ore', 'Aria', 'Tümü', etc.),
        synchronizes the UI material dropdown and canvas filter state, and returns matching node dicts.
        """
        self.set_material_filter(material_query)
        canvas = self.__dict__.get("canvas")
        if canvas is not None and hasattr(canvas, "filter_nodes_by_selected_material"):
            return canvas.filter_nodes_by_selected_material(material_query)
        cur_nodes = self.__dict__.get("_current_nodes")
        if cur_nodes is not None:
            nodes = list(cur_nodes)
        else:
            try:
                nodes, _ = self._extract_region_nodes_and_route_ids()
            except Exception:
                nodes = []
        return [
            dict(n)
            for n in nodes
            if isinstance(n, dict) and node_matches_material(n, material_query)
        ]

    def resequence_active_route_by_proximity(
        self,
        start_xy: Optional[Tuple[float, float]] = None,
        player_pos: Optional[Tuple[float, float]] = None,
    ) -> List[int]:
        """
        Resequences active route nodes starting from start_xy (or current player position)
        using Nearest-Neighbor + 2-Opt TSP optimization, persists the updated sequence to
        RouteManager if available, and returns the ordered list of node IDs.
        """
        if start_xy is None and player_pos is not None:
            start_xy = (float(player_pos[0]), float(player_pos[1]))
        p_pos = self.__dict__.get("_player_pos")
        if start_xy is None and p_pos is not None:
            start_xy = (float(p_pos[0]), float(p_pos[1]))

        ordered_ids: List[int] = []
        canvas = self.__dict__.get("canvas")
        if canvas is not None and hasattr(canvas, "resequence_active_route_by_proximity"):
            ordered_ids = canvas.resequence_active_route_by_proximity(start_xy=start_xy)
        elif self.__dict__.get("_current_nodes"):
            mat_filter = str(self.__dict__.get("_selected_material_filter") or "")
            pool = self.filter_nodes_by_selected_material(mat_filter) if mat_filter and mat_filter not in ("ALL", "Tümü") else [dict(n) for n in self.__dict__.get("_current_nodes", []) if isinstance(n, dict)]
            # S045 (v19/P3-8): başlangıç konumu yokken (0,0)'dan
            # sıralanıyordu — (0,0) harita köşesidir, oyuncu değil;
            # konum yoksa havuz sırası korunur. Ayrıca id'siz düğüme
            # `idx+1` uydurma id veriliyordu — gerçek bir id ile
            # çakışıp yanlış düğüme sıra atayabilirdi; id'siz düğüm
            # sıralamaya girmez, eski seq_index'i korunur.
            if start_xy is not None:
                sx, sy = start_xy
                pool.sort(key=lambda n: math.hypot(resolve_node_xy(n)[0] - sx, resolve_node_xy(n)[1] - sy))
            ordered_ids = []
            for n in pool:
                nid = n.get("id")
                if nid is None:
                    continue
                try:
                    ordered_ids.append(int(nid))
                except (TypeError, ValueError):
                    continue

        routes_obj = self.__dict__.get("routes")
        if ordered_ids and routes_obj is not None:
            # K-19: sıra değişimi önce bellekte yapılır; disk yazımı
            # başarısız olursa bellek eski sıraya geri alınır, yoksa
            # bellek/disk ayrışır ve hata sessizce yutulur.
            seq_snapshot: Dict[int, Any] = {}
            prev_nodes: Optional[List[Any]] = None
            prev_route_ids: Optional[List[int]] = None
            reg_entry: Optional[Dict[str, Any]] = None
            reg_data: Optional[Dict[str, Any]] = None
            try:
                reg = self.canonical_region_name(self._region_name)
                rank_map = {nid: rank for rank, nid in enumerate(ordered_ids, start=1)}

                if hasattr(self.routes, "_regions") and isinstance(self.routes._regions, dict) and reg in self.routes._regions:
                    reg_entry = self.routes._regions[reg]
                    existing_nodes = reg_entry.get("nodes", [])
                    prev_nodes = list(existing_nodes)
                    for nd in existing_nodes:
                        seq_snapshot[id(nd)] = getattr(nd, "seq_index", None) if hasattr(nd, "seq_index") else (nd.get("seq_index") if isinstance(nd, dict) else None)
                    node_by_id: Dict[int, Any] = {}
                    for nd in existing_nodes:
                        nid = getattr(nd, "id", None) or (nd.get("id") if isinstance(nd, dict) else None)
                        if nid is not None:
                            node_by_id[int(nid)] = nd

                    reordered_objs: List[Any] = []
                    for nid in ordered_ids:
                        if nid in node_by_id:
                            obj = node_by_id[nid]
                            if hasattr(obj, "seq_index"):
                                obj.seq_index = rank_map[nid]
                            if isinstance(obj, dict):
                                obj["seq_index"] = rank_map[nid]
                            reordered_objs.append(obj)

                    next_seq = len(reordered_objs) + 1
                    ordered_set = set(ordered_ids)
                    for nd in existing_nodes:
                        nid = getattr(nd, "id", None) or (nd.get("id") if isinstance(nd, dict) else None)
                        if nid is not None and int(nid) not in ordered_set:
                            if hasattr(nd, "seq_index"):
                                nd.seq_index = next_seq
                            if isinstance(nd, dict):
                                nd["seq_index"] = next_seq
                            next_seq += 1
                            reordered_objs.append(nd)

                    reg_entry["nodes"] = reordered_objs
                    if hasattr(self.routes, "save_region"):
                        saved = self.routes.save_region(reg)
                    elif hasattr(self.routes, "save_data"):
                        saved = self.routes.save_data()
                    else:
                        saved = None
                    if saved is False:
                        raise OSError("sıra diske yazılamadı (save False döndü)")

                elif hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                    reg_data = self.routes.data.get("regions", {}).get(reg, {})
                    if isinstance(reg_data, dict):
                        routes_dict = reg_data.setdefault("routes", {})
                        prev_route_ids = list(routes_dict[self._route_name]) if self._route_name in routes_dict else None
                        for nd in reg_data.get("nodes", []):
                            if isinstance(nd, dict):
                                seq_snapshot[id(nd)] = nd.get("seq_index")
                        routes_dict[self._route_name] = list(ordered_ids)
                        for nd in reg_data.get("nodes", []):
                            if isinstance(nd, dict):
                                nid = int(nd.get("id", 0))
                                if nid in rank_map:
                                    nd["seq_index"] = rank_map[nid]
                        if hasattr(self.routes, "save_data"):
                            saved = self.routes.save_data()
                        else:
                            saved = None
                        if saved is False:
                            raise OSError("sıra diske yazılamadı (save_data False döndü)")
            except Exception as exc:
                # Geri alma: bellek diskteki eski sıraya döner.
                try:
                    if reg_entry is not None and prev_nodes is not None:
                        reg_entry["nodes"] = prev_nodes
                    if reg_data is not None and isinstance(reg_data, dict):
                        routes_dict = reg_data.get("routes", {})
                        if isinstance(routes_dict, dict):
                            if prev_route_ids is None:
                                routes_dict.pop(self._route_name, None)
                            else:
                                routes_dict[self._route_name] = prev_route_ids
                    if reg_entry is not None and prev_nodes is not None:
                        for nd in prev_nodes:
                            if id(nd) in seq_snapshot and seq_snapshot[id(nd)] is not None:
                                if hasattr(nd, "seq_index"):
                                    nd.seq_index = seq_snapshot[id(nd)]
                                if isinstance(nd, dict):
                                    nd["seq_index"] = seq_snapshot[id(nd)]
                    if reg_data is not None and isinstance(reg_data, dict):
                        for nd in reg_data.get("nodes", []):
                            if isinstance(nd, dict) and id(nd) in seq_snapshot and seq_snapshot[id(nd)] is not None:
                                nd["seq_index"] = seq_snapshot[id(nd)]
                except Exception as rb_exc:
                    logger.error("resequence_active_route_by_proximity geri alma başarısız: %s", rb_exc)
                logger.error("resequence_active_route_by_proximity kalıcılık hatası (%s): %s; bellek eski sıraya alındı.", self._region_name, exc)

        return ordered_ids

    def _on_material_filter_changed(self, text: str) -> None:
        """Slot for combo material filter change."""
        val = self.get_material_filter()
        self.canvas.set_material_filter(val)

    def _on_draw_proximity_route_clicked(self) -> None:
        """Slot for draw proximity route button."""
        self.draw_proximity_route()

    def _populate_material_filter_combo(self) -> None:
        """Dynamically populates material filter combo with distinct materials from current nodes."""
        if not hasattr(self, "combo_material_filter") or self.combo_material_filter is None:
            return
        existing = {
            normalize_material_token(str(self.combo_material_filter.itemData(i) or self.combo_material_filter.itemText(i)))
            for i in range(self.combo_material_filter.count())
        }
        for node in getattr(self.canvas, "_nodes", []):
            if isinstance(node, dict):
                rtype = str(node.get("resource_type") or node.get("material_type") or node.get("name") or "").strip()
                norm = normalize_material_token(rtype)
                if rtype and norm not in existing and len(norm) > 2:
                    self.combo_material_filter.addItem(rtype, userData=rtype)
                    existing.add(norm)

    def set_region(self, region_name: str, route_name: Optional[str] = None) -> None:
        canon_reg = self.canonical_region_name(region_name)
        if route_name is None:
            if str(region_name).strip().lower() != canon_reg.lower():
                route_name = str(region_name).strip()
            else:
                route_name = self._route_name or "All"
        self._region_name = canon_reg
        self._route_name = route_name
        self.reload_map_and_nodes()

    def set_route(self, route_name: str) -> None:
        self._route_name = route_name or "All"
        self.reload_map_and_nodes()


    def set_player_position(self, x: Optional[float], y: Optional[float], angle_deg: float = 0.0) -> None:
        # Keep the heading: the canvas stores a 3-tuple (x, y, angle) and the
        # old 2-tuple dropped the angle, resetting it on every reload.
        self._player_pos = (x, y, float(angle_deg)) if (x is not None and y is not None) else None
        self.canvas.set_player_position(x, y, angle_deg)

    def get_dry_run_trajectory_points(self) -> List[Tuple[float, float]]:
        """
        Returns simulated trajectory points from DryRunSimulator (wrapped in try/except, returning [] if none).
        """
        try:
            from core.dry_run_simulator import DryRunReplaySimulator
            simulator = DryRunReplaySimulator()
            nodes, _ = self._extract_region_nodes_and_route_ids()
            if not nodes:
                nodes = list(self.canvas._nodes)
            if not nodes:
                return []
            audit_result = simulator.run_8x_route_audit(nodes)
            # Extract coordinates from audit segments or nodes
            trajectory: List[Tuple[float, float]] = []
            segments = audit_result.get("segments", [])
            if segments:
                for seg in segments:
                    p1 = seg.get("p1")
                    if p1 and isinstance(p1, (list, tuple)) and len(p1) >= 2:
                        trajectory.append((float(p1[0]), float(p1[1])))
                # Append final point of last segment
                if segments:
                    last_p2 = segments[-1].get("p2")
                    if last_p2 and isinstance(last_p2, (list, tuple)) and len(last_p2) >= 2:
                        trajectory.append((float(last_p2[0]), float(last_p2[1])))
            else:
                for n in nodes:
                    xy = simulator._extract_xy(n)
                    trajectory.append((float(xy[0]), float(xy[1])))
            return trajectory
        except Exception as exc:
            logger.debug("get_dry_run_trajectory_points error or none available: %s", exc)
            return []

    # -------------------------------------------------------------------------
    # Interactive Node Toggling & Context Menu (Türkçe)
    # -------------------------------------------------------------------------
    def _on_node_left_clicked(self, node_id: int) -> None:
        """Left-clicking a node badge toggles its membership in the active route."""
        try:
            self._toggle_node_in_active_route(node_id)
            self.node_toggled.emit(node_id)
        except Exception as exc:
            logger.error("Error toggling node %d: %s", node_id, exc)

    def _toggle_node_in_active_route(self, node_id: int) -> None:
        try:
            reg = self.canonical_region_name(self._region_name)
            self._region_name = reg
            if self.routes is not None:
                if hasattr(self.routes, "_regions") and isinstance(self.routes._regions, dict) and reg in self.routes._regions:
                    reg_info = self.routes._regions[reg]
                    target_node = next(
                        (n for n in reg_info.get("nodes", [])
                         if int(getattr(n, "id", -1) if hasattr(n, "id") else (n.get("id", -1) if isinstance(n, dict) else -1)) == int(node_id)),
                        None
                    )
                    if target_node is not None:
                        if self._route_name == "All":
                            # 'All' is a virtual route: no node carries it as a
                            # real membership entry, so toggling the routes list
                            # was a no-op. Membership in 'All' is the in_route
                            # flag instead.
                            if isinstance(target_node, dict):
                                cur = target_node.get("in_route", True)
                                target_node["in_route"] = not cur
                                routes = target_node.setdefault("routes", [])
                                if cur is not False:
                                    if "All" in routes:
                                        routes.remove("All")
                                else:
                                    if "All" not in routes:
                                        routes.append("All")
                            else:
                                cur = getattr(target_node, "in_route", True)
                                target_node.in_route = not cur
                                routes = getattr(target_node, "routes", None)
                                if isinstance(routes, list):
                                    if cur is not False:
                                        if "All" in routes:
                                            routes.remove("All")
                                    elif "All" not in routes:
                                        routes.append("All")
                            if cur is not False:
                                logger.info("Nokta #%d '%s' (%s) rotasından çıkarıldı", node_id, self._route_name, reg)
                            else:
                                logger.info("Nokta #%d '%s' (%s) rotasına eklendi", node_id, self._route_name, reg)
                        else:
                            routes = target_node.routes if hasattr(target_node, "routes") else (target_node.setdefault("routes", []) if isinstance(target_node, dict) else [])
                            if self._route_name in routes:
                                routes.remove(self._route_name)
                                logger.info("Nokta #%d '%s' (%s) rotasından çıkarıldı", node_id, self._route_name, reg)
                            else:
                                routes.append(self._route_name)
                                logger.info("Nokta #%d '%s' (%s) rotasına eklendi", node_id, self._route_name, reg)
                    if hasattr(self.routes, "save_region"):
                        self.routes.save_region(reg)
                    elif hasattr(self.routes, "save_data"):
                        self.routes.save_data()
                elif hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                    regions = self.routes.data.get("regions", {})
                    reg_data = regions.get(reg, {})
                    routes_dict = reg_data.setdefault("routes", {})
                    active_list: List[int] = routes_dict.setdefault(self._route_name, [])

                    if int(node_id) in active_list:
                        active_list.remove(int(node_id))
                        logger.info("Nokta #%d '%s' (%s) rotasından çıkarıldı", node_id, self._route_name, reg)
                    else:
                        active_list.append(int(node_id))
                        logger.info("Nokta #%d '%s' (%s) rotasına eklendi", node_id, self._route_name, reg)

                    if hasattr(self.routes, "save_data"):
                        self.routes.save_data()
        except Exception as exc:
            logger.error("Error in _toggle_node_in_active_route: %s", exc)

        nodes, route_ids = self._extract_region_nodes_and_route_ids()
        self.canvas.set_nodes_data(nodes, route_ids, rest_duration_sec=self._rest_duration_sec)

    def _on_node_right_clicked(self, node_id: int, global_pos: QPoint) -> None:
        """Sağ tıklandığında nokta detaylarını, hasat durumunu (0/3..3x) ve Türkçe işlemleri açar."""
        try:
            nodes, route_ids = self._extract_region_nodes_and_route_ids()
            node_obj = next((n for n in nodes if int(n.get("id", -1)) == node_id), None)
            if node_obj is None:
                return

            in_route = node_id in route_ids
            nx = node_obj.get("map_x", node_obj.get("x", 0))
            ny = node_obj.get("map_y", node_obj.get("y", 0))
            visits = node_obj.get("visit_count", 0)
            badge_str = _MapCanvas._format_harvest_badge(node_obj)
            is_auto = _MapCanvas._is_minimap_auto_node(node_obj)
            src_label = "Minimap/3B Oto-Kayıt" if is_auto else "Manuel Kayıt"

            menu = QMenu(self)
            menu.setStyleSheet("""
                QMenu {
                    background-color: #181b20;
                    color: #e5e7eb;
                    border: 1px solid #2d323b;
                    border-radius: 6px;
                    padding: 4px;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 13px;
                }
                QMenu::item {
                    padding: 8px 22px 8px 14px;
                    border-radius: 4px;
                }
                QMenu::item:selected {
                    background-color: #2563eb;
                    color: #ffffff;
                }
                QMenu::item:disabled {
                    color: #9ca3af;
                    font-weight: bold;
                }
                QMenu::separator {
                    height: 1px;
                    background: #2d323b;
                    margin: 6px 8px;
                }
            """)

            header_act = QAction(
                f"Nokta #{node_id}  ({nx}, {ny}) · Hasat: {badge_str} · {src_label} · Ziyaret: {visits}x",
                menu,
            )
            header_act.setEnabled(False)
            menu.addAction(header_act)
            menu.addSeparator()

            toggle_label = (
                "Aktif Rotadan Çıkar (Rota Dışı / Pasif Yap)"
                if in_route
                else "Aktif Rotaya Ekle (Aktif Yap)"
            )
            act_toggle = QAction(toggle_label, menu)
            act_toggle.triggered.connect(lambda: self._on_node_left_clicked(node_id))
            menu.addAction(act_toggle)

            act_ready = QAction("Durumu 'Aktif' Yap (Bekleme Süresini Sıfırla)", menu)
            act_ready.triggered.connect(lambda: self._set_node_cooldown(node_id, resting=False))
            menu.addAction(act_ready)

            act_rest = QAction("Durumu 'Dinleniyor (5dk)' Yap", menu)
            act_rest.triggered.connect(lambda: self._set_node_cooldown(node_id, resting=True))
            menu.addAction(act_rest)

            menu.addSeparator()

            for h_idx, h_label in ((0, "0/3"), (1, "1/3"), (2, "2/3"), (3, "3x")):
                act_h = QAction(f"Hasat Rozetini '{h_label}' Olarak Ayarla", menu)
                act_h.triggered.connect(
                    lambda _checked=False, hc=h_idx, hb=h_label: self._set_node_harvest_status(
                        node_id, hc, max_harvests=3, badge_text=hb
                    )
                )
                menu.addAction(act_h)

            menu.addSeparator()

            act_delete = QAction(f"Noktayı Sil (#{node_id})", menu)
            act_delete.triggered.connect(lambda: self._delete_node(node_id))
            menu.addAction(act_delete)

            menu.popup(global_pos)
        except Exception as exc:
            logger.error("Failed opening node context menu for #%d: %s", node_id, exc)

    def _set_node_harvest_status(
        self,
        node_id: int,
        harvest_count: int,
        max_harvests: int = 3,
        badge_text: Optional[str] = None,
    ) -> None:
        """Noktanın hasat durum rozetini ('0/3', '1/3', '2/3' veya '3x') günceller."""
        try:
            reg = self.canonical_region_name(self._region_name)
            self._region_name = reg
            if self.routes is not None:
                if hasattr(self.routes, "_regions") and isinstance(self.routes._regions, dict) and reg in self.routes._regions:
                    reg_info = self.routes._regions[reg]
                    for n in reg_info.get("nodes", []):
                        nid = getattr(n, "id", None) or (n.get("id") if isinstance(n, dict) else None)
                        if nid is not None and int(nid) == int(node_id):
                            if hasattr(n, "current_harvests"):
                                n.current_harvests = int(harvest_count)
                                n.max_harvests = int(max_harvests)
                            if isinstance(n, dict):
                                n["current_harvests"] = int(harvest_count)
                                n["max_harvests"] = int(max_harvests)
                                n["harvest_count"] = int(harvest_count)
                            if badge_text:
                                if hasattr(n, "harvest_badge"):
                                    n.harvest_badge = str(badge_text)
                                if isinstance(n, dict):
                                    n["harvest_badge"] = str(badge_text)
                            break
                    if hasattr(self.routes, "save_region"):
                        self.routes.save_region(reg)
                    elif hasattr(self.routes, "save_data"):
                        self.routes.save_data()
                elif hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                    reg_data = self.routes.data.get("regions", {}).get(reg, {})
                    for n in reg_data.get("nodes", []):
                        if isinstance(n, dict) and int(n.get("id", -1)) == int(node_id):
                            n["harvest_count"] = int(harvest_count)
                            n["max_harvests"] = int(max_harvests)
                            if badge_text:
                                n["harvest_badge"] = str(badge_text)
                            break
                    if hasattr(self.routes, "save_data"):
                        self.routes.save_data()
            for n in self.canvas._nodes:
                if isinstance(n, dict) and int(n.get("id", -1)) == int(node_id):
                    n["harvest_count"] = int(harvest_count)
                    n["max_harvests"] = int(max_harvests)
                    if badge_text:
                        n["harvest_badge"] = str(badge_text)
                    break
            self.canvas.update()
        except Exception as exc:
            logger.error("Hasat durumu güncellenemedi (#%d): %s", node_id, exc)

    def _set_node_cooldown(self, node_id: int, resting: bool) -> None:
        try:
            reg = self.canonical_region_name(self._region_name)
            self._region_name = reg
            if self.routes is not None:
                if hasattr(self.routes, "_regions") and isinstance(self.routes._regions, dict) and reg in self.routes._regions:
                    reg_info = self.routes._regions[reg]
                    for n in reg_info.get("nodes", []):
                        nid = getattr(n, "id", None) or (n.get("id") if isinstance(n, dict) else None)
                        if nid is not None and int(nid) == int(node_id):
                            now_ts = time.time() if resting else 0.0
                            new_state = "resting" if resting else "active"
                            if hasattr(n, "last_visited_timestamp"):
                                n.last_visited_timestamp = now_ts
                                n.state = new_state
                            if isinstance(n, dict):
                                n["last_visited"] = now_ts
                                n["last_visited_timestamp"] = now_ts
                                n["status"] = "resting" if resting else "ready"
                                n["state"] = new_state
                            break
                    if hasattr(self.routes, "save_region"):
                        self.routes.save_region(reg)
                    elif hasattr(self.routes, "save_data"):
                        self.routes.save_data()
                elif hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                    reg_data = self.routes.data.get("regions", {}).get(reg, {})
                    for n in reg_data.get("nodes", []):
                        if int(n.get("id", -1)) == int(node_id):
                            # S043 (v19/P3-8): bu dal yalnızca
                            # last_visited+status yazıyordu — dict'te
                            # last_visited_timestamp ayrı anahtardır
                            # (NodeData'daki property takma adının aksine),
                            # bayat kalıyordu. Diğer dallarla aynı alanlar.
                            n["last_visited"] = time.time() if resting else 0
                            n["last_visited_timestamp"] = time.time() if resting else 0.0
                            n["status"] = "resting" if resting else "ready"
                            n["state"] = "resting" if resting else "active"
                            break
                    if hasattr(self.routes, "save_data"):
                        self.routes.save_data()
            nodes, route_ids = self._extract_region_nodes_and_route_ids()
            self.canvas.set_nodes_data(nodes, route_ids, rest_duration_sec=self._rest_duration_sec)
        except Exception as exc:
            logger.error("Failed updating cooldown state for node #%d: %s", node_id, exc)

    def delete_node(self, node_id: int) -> bool:
        """Public API to delete node from route manager and reload map canvas.

        Returns True only if the node was actually removed.
        """
        return self._delete_node(node_id)

    def _delete_node(self, node_id: int) -> bool:
        try:
            reg = self.canonical_region_name(self._region_name)
            self._region_name = reg
            removed = False
            if self.routes is not None:
                # One authoritative delete: RouteManager.delete_node already
                # removes the node AND persists; re-filtering + re-saving the
                # region afterwards would delete/save it a second time.
                if hasattr(self.routes, "delete_node") and callable(self.routes.delete_node):
                    try:
                        res = self.routes.delete_node(node_id, region=reg)
                    except TypeError:
                        res = self.routes.delete_node(node_id)
                    removed = True if res is None else bool(res)
                if not removed and hasattr(self.routes, "_regions") and isinstance(self.routes._regions, dict) and reg in self.routes._regions:
                    reg_info = self.routes._regions[reg]
                    if "nodes" in reg_info:
                        before = len(reg_info["nodes"])
                        reg_info["nodes"] = [
                            n for n in reg_info["nodes"]
                            if int(getattr(n, "id", -1) if hasattr(n, "id") else (n.get("id", -1) if isinstance(n, dict) else -1)) != int(node_id)
                        ]
                        removed = len(reg_info["nodes"]) < before
                        for idx, n in enumerate(reg_info["nodes"], start=1):
                            if hasattr(n, "seq_index"):
                                n.seq_index = idx
                            if isinstance(n, dict):
                                n["seq_index"] = idx

                    if removed:
                        if hasattr(self.routes, "save_region"):
                            self.routes.save_region(reg)
                        elif hasattr(self.routes, "save_data"):
                            self.routes.save_data()
                elif not removed and hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                    reg_data = self.routes.data.get("regions", {}).get(reg, {})
                    if "nodes" in reg_data:
                        before = len(reg_data["nodes"])
                        reg_data["nodes"] = [
                            n for n in reg_data["nodes"] if int(n.get("id", -1)) != int(node_id)
                        ]
                        removed = len(reg_data["nodes"]) < before
                        for idx, n in enumerate(reg_data["nodes"], start=1):
                            if isinstance(n, dict):
                                n["seq_index"] = idx
                    for r_name, r_list in reg_data.get("routes", {}).items():
                        if int(node_id) in r_list:
                            r_list.remove(int(node_id))
                    if removed and hasattr(self.routes, "save_data"):
                        self.routes.save_data()

            if not removed:
                logger.warning("Silinecek düğüm bulunamadı: #%d (%s)", node_id, reg)
                return False

            nodes, route_ids = self._extract_region_nodes_and_route_ids()
            self.canvas.set_nodes_data(nodes, route_ids, rest_duration_sec=self._rest_duration_sec)
            self.node_deleted.emit(int(node_id))
            logger.info("Deleted node #%d from canonical region '%s'", node_id, reg)
            return True
        except Exception as exc:
            logger.error("Failed deleting node #%d: %s", node_id, exc)
            return False

    def get_map_viewer_route_summary(self) -> Dict[str, Any]:
        """Returns a structured summary of the currently loaded region, route, and filtered nodes.

        Prefers the live window state (_region_name / RouteManager nodes);
        falls back to the legacy _current_region / _current_nodes dict entries
        (used by lightweight / uninitialised dialog harnesses) when present.
        """
        d = self.__dict__
        region = d.get("_current_region") or d.get("_region_name") or "Altgard"
        route = d.get("_current_route") or d.get("_route_name") or "All"
        nodes = list(d.get("_current_nodes") or [])
        if not nodes:
            try:
                nodes, _route_ids = self._extract_region_nodes_and_route_ids()
            except Exception:
                nodes = []
        if not nodes:
            canvas = d.get("canvas")
            if canvas is not None:
                try:
                    nodes = [dict(n) for n in getattr(canvas, "_nodes", []) if isinstance(n, dict)]
                except Exception:
                    nodes = []
        mat_filter = str(d.get("_selected_material_filter") or "Tümü")
        if not mat_filter or mat_filter in ("Tümü", "ALL", "All"):
            filtered = list(nodes)
        else:
            filtered = [n for n in nodes if node_matches_material(n, mat_filter)]
        # S-1803 / Decision 3: bind the ready flag to the real contract
        # probe plus live state — a pixmap-less / empty canvas is not
        # 'ready'. A bare dialog harness with no canvas attached (used by
        # lightweight tests) has no pixmap to check, so only the node
        # count applies there.
        canvas = d.get("canvas") or d.get("_canvas")
        has_pixmap = True
        if canvas is not None:
            pix = getattr(canvas, "_base_pixmap", None)
            has_pixmap = pix is not None and not pix.isNull()
        ready = bool(validate_map_viewer_contract() and nodes and has_pixmap)
        return {
            "region": str(region),
            "route": str(route),
            "total_nodes": len(nodes),
            "filtered_nodes": len(filtered),
            "selected_material": mat_filter,
            "ready": ready,
        }

    def get_map_viewer_health_summary(self) -> dict:
        try:
            d = self.__dict__
            canvas = d.get("canvas")
            node_count = 0
            if canvas is not None:
                try:
                    node_count = len(getattr(canvas, "_nodes", []) or [])
                except Exception:
                    node_count = 0
            return {
                "map_viewer_ready": validate_map_viewer_contract(),
                "canvas_ready": canvas is not None,
                "region": str(d.get("_region_name") or ""),
                "route": str(d.get("_route_name") or ""),
                "node_count": node_count,
            }
        except Exception:
            return {"map_viewer_ready": False}

    def get_map_viewer_ready_flag(self) -> dict:
        return {"module": "map_viewer", "ready": validate_map_viewer_contract()}

    def get_map_viewer_version_info(self) -> dict:
        return {"module": "map_viewer", "version": __version__, "ready": validate_map_viewer_contract()}

    def get_map_viewer_diagnostic_snapshot(self) -> dict:
        return {"module": "map_viewer", "status": "ok" if validate_map_viewer_contract() else "degraded", "ready": validate_map_viewer_contract()}


# Backwards compatibility alias so existing imports (`from ui.map_viewer import MapViewerDialog`) continue to work
MapViewerDialog = MapViewerWindow
MapViewer = MapViewerWindow


def get_map_viewer_capabilities() -> list:
    return ["map_viewer", "route_visualization", "diagnostic_snapshot"]


def validate_map_viewer_contract() -> bool:
    """Actually checks the map viewer contract instead of rubber-stamping it.

    Verifies the capability list is non-empty, both viewer classes exist, and
    the public methods the rest of the app calls are present and callable.
    """
    try:
        caps = get_map_viewer_capabilities()
        if not caps:
            return False
        if MapViewerWindow is None or _MapCanvas is None:
            return False
        window_methods = (
            "reload_map_and_nodes", "add_node", "move_node", "delete_node",
            "optimize_sequential_route", "set_material_filter",
            "set_player_position", "get_map_viewer_route_summary",
        )
        if not all(callable(getattr(MapViewerWindow, m, None)) for m in window_methods):
            return False
        canvas_methods = (
            "set_nodes_data", "set_map_pixmap", "set_player_position",
            "calculate_proximity_route", "delete_node",
        )
        if not all(callable(getattr(_MapCanvas, m, None)) for m in canvas_methods):
            return False
        return True
    except Exception:
        return False


def get_map_viewer_baseline_metrics() -> dict:
    return {"module": "map_viewer", "contract_valid": validate_map_viewer_contract(), "capability_count": len(get_map_viewer_capabilities())}


def get_map_viewer_readiness_status() -> dict:
    return {"module": "map_viewer", "ready": validate_map_viewer_contract(), "metrics": get_map_viewer_baseline_metrics()}


def get_map_viewer_audit_record() -> dict:
    return {"module": "map_viewer", "audit_passed": validate_map_viewer_contract(), "readiness": get_map_viewer_readiness_status()}


def get_map_viewer_operational_digest() -> dict:
    return {"module": "map_viewer", "operational": validate_map_viewer_contract(), "audit": get_map_viewer_audit_record()}


def get_map_viewer_telemetry_payload() -> dict:
    return {"module": "map_viewer", "telemetry_ready": validate_map_viewer_contract(), "digest": get_map_viewer_operational_digest()}


def get_map_viewer_integrity_seal() -> dict:
    return {"module": "map_viewer", "sealed": validate_map_viewer_contract(), "telemetry": get_map_viewer_telemetry_payload()}


def get_map_viewer_compliance_certificate() -> dict:
    return {"module": "map_viewer", "certified": validate_map_viewer_contract(), "seal": get_map_viewer_integrity_seal()}


def get_map_viewer_readiness_probe() -> dict:
    return {"module": "map_viewer", "ready": validate_map_viewer_contract(), "certificate": get_map_viewer_compliance_certificate()}


def get_map_viewer_audit_summary() -> dict:
    return {"module": "map_viewer", "audit_passed": validate_map_viewer_contract(), "probe": get_map_viewer_readiness_probe()}

