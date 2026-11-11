"""R24 düzeltme testleri (F-1642, S-1301) — core/route_manager.py.

F-1642: _send_map_key, press_key'in döndürdüğü float bekleme süresini
  (0.0 = gönderim iptal) doğru yorumlamalı; 0.0 artık True dönmemeli.
S-1301: delete_region, rmtree yarıda hata verirse klasörü + bellek
  durumunu geri yüklemeli (snapshot geri alma).
"""

import os

import pytest

from core.route_manager import RouteManager


class _FakeInput:
    def __init__(self, result):
        self._result = result

    def press_key(self, key):
        return self._result


@pytest.fixture()
def manager(tmp_path):
    mgr = RouteManager(base_dir=str(tmp_path / "routes"))
    assert mgr.create_region("TestRegion")
    return mgr


# --- F-1642: _send_map_key -----------------------------------------------

def test_send_map_key_press_key_zero_is_failure():
    mgr = RouteManager(base_dir="routes")
    assert mgr._send_map_key(_FakeInput(0.0)) is False


def test_send_map_key_press_key_positive_float_is_success():
    mgr = RouteManager(base_dir="routes")
    assert mgr._send_map_key(_FakeInput(0.065)) is True


def test_send_map_key_bool_results_unchanged():
    mgr = RouteManager(base_dir="routes")
    assert mgr._send_map_key(_FakeInput(True)) is True
    assert mgr._send_map_key(_FakeInput(False)) is False


def test_send_map_key_none_result_still_assumed_sent():
    mgr = RouteManager(base_dir="routes")
    assert mgr._send_map_key(_FakeInput(None)) is True


def test_send_map_key_no_input_ctrl_is_failure():
    mgr = RouteManager(base_dir="routes")
    assert mgr._send_map_key(None) is False


# --- S-1301: delete_region ------------------------------------------------

def test_delete_region_success_removes_dir_and_memory(manager, tmp_path):
    reg_dir = tmp_path / "routes" / "TestRegion"
    assert reg_dir.is_dir()
    assert manager.delete_region("TestRegion") is True
    assert "TestRegion" not in manager.get_regions()
    assert not reg_dir.exists()
    assert not os.path.exists(str(reg_dir) + f".siliniyor-{os.getpid()}")


def test_delete_region_unknown_returns_false(manager):
    assert manager.delete_region("YokBolge") is False


def test_delete_region_rmtree_failure_rolls_back(manager, tmp_path, monkeypatch):
    reg_dir = tmp_path / "routes" / "TestRegion"
    sentinel = reg_dir / "region_data.json"
    assert sentinel.is_file()
    sentinel_content = sentinel.read_bytes()
    manager.active_region = "TestRegion"

    real_rmtree = __import__("shutil").rmtree

    def failing_rmtree(path, *args, **kwargs):
        if "siliniyor" in str(path):
            raise OSError("simüle rmtree hatası")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("core.route_manager.shutil.rmtree", failing_rmtree)
    assert manager.delete_region("TestRegion") is False
    # Geri alma: bellek durumu + klasör (içeriğiyle) yerinde durmalı.
    assert "TestRegion" in manager.get_regions()
    assert manager.active_region == "TestRegion"
    assert reg_dir.is_dir()
    assert sentinel.is_file()
    assert sentinel.read_bytes() == sentinel_content
    assert not os.path.exists(str(reg_dir) + f".siliniyor-{os.getpid()}")
