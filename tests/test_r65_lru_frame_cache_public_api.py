# -*- coding: utf-8 -*-
"""R65 (F-2000): `LRUFrameDecodeCache` public-API wiring tests.

F-2000: `LRUFrameDecodeCache` / `get_lru_frame_cache` were only referenced by
`tests/test_w61_lru_frame_cache_performance.py` — no production module imported
them and `dataset_factory/__init__.py` did not export them.

Wiring split (per the R62 / F-1980 precedent):
  * package-level export in `dataset_factory/__init__.py` — slice R46;
  * Step 2 consumer wiring in `ui/yolo_factory_dialog.py` — slice R44;
  * this file (R65) proves the wiring from the module side: the module API
    exists and is importable, and the package export exists once R46 lands.

`test_package_exports_lru_frame_cache` therefore FAILS until R46 adds the
export — that failure is the finding, not a test bug.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import dataset_factory
from dataset_factory import lru_frame_cache as lru_module
from dataset_factory.lru_frame_cache import (
    DEFAULT_CACHE_CAPACITY,
    FrameCacheEntry,
    LRUFrameDecodeCache,
    get_lru_frame_cache,
    imread_unicode,
)


def test_module_public_api_importable_and_singleton() -> None:
    """Module-level public API exists, `__all__` lists it, shared cache is a singleton."""
    assert DEFAULT_CACHE_CAPACITY == 32
    for name in (
        "DEFAULT_CACHE_CAPACITY",
        "FrameCacheEntry",
        "LRUFrameDecodeCache",
        "get_lru_frame_cache",
        "imread_unicode",
    ):
        assert name in lru_module.__all__
        assert getattr(lru_module, name) is not None
    first = get_lru_frame_cache()
    second = get_lru_frame_cache()
    assert first is second
    assert isinstance(first, LRUFrameDecodeCache)
    assert first.capacity == DEFAULT_CACHE_CAPACITY


def test_package_exports_lru_frame_cache() -> None:
    """`dataset_factory` package exports the cache API (R46 export; F-2000)."""
    assert "LRUFrameDecodeCache" in dataset_factory.__all__
    assert "get_lru_frame_cache" in dataset_factory.__all__
    assert "imread_unicode" in dataset_factory.__all__
    assert dataset_factory.LRUFrameDecodeCache is LRUFrameDecodeCache
    assert dataset_factory.get_lru_frame_cache is get_lru_frame_cache
    assert dataset_factory.imread_unicode is imread_unicode
    assert dataset_factory.get_lru_frame_cache() is get_lru_frame_cache()
