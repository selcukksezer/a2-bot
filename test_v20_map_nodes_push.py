"""v20/map-nodes regresyon testleri (KOŞULMADI — VM'de pytest yasak; PC'de koşulacak).

Madde 5: oto-haritalanan düğümler (#288/#289/#290, Targena) rota verisine
girmesine rağmen haritada görünmüyordu.

Kök neden (kod kanıtı):
- `ExplorerModeController.register_map_viewer` üretimde HİÇ çağrılmıyordu
  (yalnızca test_worker_mr6_map_radar.py çağırıyordu) — dolayısıyla
  `AutoCartographer._emit_node_discovered` / `ExplorerMode._emit_node_discovered`
  harita penceresine hiç ulaşmıyordu; harita yalnızca 1 sn'lik poll'e
  (shell sekmesi) veya manuel yenilemeye bağlıydı. Bağımsız harita
  penceresine (main.open_map_viewer) `set_auto_cartographer` bile
  yapılmıyordu — poll'u kartografsız no-op'tu.
- `SequentialRoutePlanner` içindeki norm->piksel çıkarmaları boyutsuz
  çağrılıyordu; "Norm-koordinat dönüşümünde boyut verilmedi" uyarısı
  verip sessizce 1280x720 varsayıyordu.

Bu testler:
1. Router'ın norm çıkarmalarının artık kanvas boyutunu açıkça geçirdiğini
   (uyarı çıkmadığını ve 1280x720 ölçeğinin kullanıldığını) doğrular.
2. Shell Harita sayfasının (`_wire_map_page`) pencereyi ExplorerMode'a
   dinleyici olarak kaydettiğini doğrular.
"""

import logging
import types
from unittest.mock import MagicMock

import pytest

import core.sequential_router as seq_router
import core.route_manager as route_manager_mod


def _norm_only_node(node_id, nx, ny):
    """Yalnızca norm koordinatı olan düğüm (map_x/map_y yok)."""
    return {"id": node_id, "norm_x": nx, "norm_y": ny, "resource_type": "Targena"}


def test_router_extraction_uses_canvas_size_without_fallback_warning(caplog):
    """build_sequential_order / insert_node_in_sequence norm düğümlerinde
    'boyut verilmedi' uyarısı çıkarmaz ve norm'u 1280x720 ile çevirir."""
    # Bir kezlik uyarı bayrakları süreçte kalıcıdır — testten önce sıfırla.
    seq_router._NORM_CANVAS_FALLBACK_WARNED = False
    route_manager_mod._NORM_CANVAS_FALLBACK_WARNED = False

    planner = seq_router.SequentialRoutePlanner()
    nodes = [
        _norm_only_node(288, 0.5214, 0.4926),
        _norm_only_node(289, 0.4766, 0.4935),
        _norm_only_node(290, 0.4969, 0.4917),
    ]

    with caplog.at_level(logging.WARNING, logger="core.sequential_router"):
        with caplog.at_level(logging.WARNING, logger="core.route_manager"):
            ordered = planner.build_sequential_order(list(nodes), mode="proximity")
            ordered, _idx = planner.insert_node_in_sequence(
                ordered, _norm_only_node(291, 0.6, 0.6)
            )
            # Doğrudan birim kontrolü: norm (0.25, 0.5) -> (320, 360).
            # Boyut açıkça geçirilir (yardımcının boyutsuz çağrısı
            # tasarım gereği bir kez uyarır — test edilen sözleşme
            # çağrı yerlerinin boyutu açıkça geçirmesidir).
            assert seq_router._extract_xy_or_none(
                _norm_only_node(999, 0.25, 0.5),
                fallback_width=seq_router.MAP_CANVAS_WIDTH,
                fallback_height=seq_router.MAP_CANVAS_HEIGHT,
            ) == (320.0, 360.0)

    assert len(ordered) == 4
    assert sorted(int(n.get("seq_index", 0)) for n in ordered) == [1, 2, 3, 4]
    assert not any(
        "boyut verilmedi" in (rec.getMessage() or "")
        for rec in caplog.records
    ), [rec.getMessage() for rec in caplog.records]


def test_wire_map_page_registers_viewer_with_explorer():
    """Shell Harita sayfası oluşturulunca pencere ExplorerMode'a
    dinleyici olarak kaydedilir (düğüm keşfinde reload tetiklenir)."""
    pytest.importorskip("PyQt6")
    from ui.app_shell import GatherionShell

    cartographer = MagicMock()
    explorer = MagicMock()
    widget = MagicMock()

    fake_self = types.SimpleNamespace(
        _auto_cartographer=cartographer,
        _explorer=explorer,
        _on_map_node_toggled=MagicMock(),
        _on_map_node_deleted=MagicMock(),
        _sync_region_route_to_page=MagicMock(),
    )

    GatherionShell._wire_map_page(fake_self, widget)

    widget.set_auto_cartographer.assert_called_once_with(cartographer)
    explorer.register_map_viewer.assert_called_once_with(widget)


def test_wire_map_page_tolerates_missing_explorer():
    """Explorer bağlı değilse _wire_map_page patlamaz (eski davranış)."""
    pytest.importorskip("PyQt6")
    from ui.app_shell import GatherionShell

    cartographer = MagicMock()
    widget = MagicMock()
    fake_self = types.SimpleNamespace(
        _auto_cartographer=cartographer,
        _explorer=None,
        _on_map_node_toggled=MagicMock(),
        _on_map_node_deleted=MagicMock(),
        _sync_region_route_to_page=MagicMock(),
    )

    GatherionShell._wire_map_page(fake_self, widget)

    widget.set_auto_cartographer.assert_called_once_with(cartographer)
