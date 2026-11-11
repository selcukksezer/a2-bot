"""R45 / F-0643 regresyon testleri.

reload_map_and_nodes() düğümlerden sin/cos salınımlı sentetik yürüyüş izi
üretmemeli ve oyuncu konumunu sentetik izin son noktasından (sabit 45°
bakış açısıyla) uydurmamalıdır. Yürüyüş izi yalnızca kaydedilmiş gerçek
oyuncu hareketidir; oyuncu konumu yalnızca canlı kareden gelir.
"""

from PyQt6.QtWidgets import QApplication

from ui.cartography_radar import CartographyRadarDialog

_NODES = [
    {"id": 1, "map_x": 100, "map_y": 100, "resource_type": "Ore"},
    {"id": 2, "map_x": 300, "map_y": 120, "resource_type": "Ore"},
    {"id": 3, "map_x": 500, "map_y": 300, "resource_type": "Herb"},
]


def _dialog_with_nodes():
    app = QApplication.instance() or QApplication([])
    dialog = CartographyRadarDialog(route_manager=None)
    app.processEvents()
    dialog.set_sequential_nodes(list(_NODES))
    return app, dialog


def test_generate_default_walk_trajectory_returns_empty_for_nodes():
    app, dialog = _dialog_with_nodes()
    try:
        assert dialog._generate_default_walk_trajectory(list(_NODES), dialog.canvas._base_pixmap) == []
    finally:
        dialog.close()


def test_reload_does_not_fabricate_trajectory_or_player_position():
    app, dialog = _dialog_with_nodes()
    try:
        assert dialog.canvas._player_pos is None
        dialog.reload_map_and_nodes()
        app.processEvents()
        assert dialog._walk_trajectory == []
        assert dialog.canvas._walk_trajectory == []
        # Oyuncu konumu düğümlerden uydurulmaz: bilinmiyorsa None kalır.
        assert dialog.canvas._player_pos is None
        assert len(dialog._ordered_nodes) == 3
    finally:
        dialog.close()


def test_reload_preserves_real_trajectory_and_real_player_position():
    app, dialog = _dialog_with_nodes()
    try:
        real_traj = [(110.0, 105.0), (210.0, 110.0), (310.0, 115.0)]
        dialog.set_walk_trajectory(real_traj)
        dialog.canvas.set_player_position(310.0, 115.0, angle_deg=30.0)
        dialog.reload_map_and_nodes()
        app.processEvents()
        assert dialog._walk_trajectory == real_traj
        assert dialog.canvas._walk_trajectory == real_traj
        assert dialog.canvas._player_pos == (310.0, 115.0, 30.0)
    finally:
        dialog.close()
