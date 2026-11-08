# -*- coding: utf-8 -*-
"""W61: `LRUFrameDecodeCache` kapasite sınırı (LRU eviction), `mtime` değişiminde
otomatik önbellek yenileme ve önbellek isabet oranı / performans doğrulama testleri.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataset_factory.lru_frame_cache import (
    DEFAULT_CACHE_CAPACITY,
    LRUFrameDecodeCache,
    imread_unicode,
)


def _write_test_image(path: Path, fill_value: int, shape: tuple[int, int, int] = (32, 48, 3)) -> np.ndarray:
    """Belirtilen piksel değeriyle doldurulmuş PNG görüntüsünü diske yazar."""
    img = np.full(shape, fill_value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", img)
    assert ok is True
    path.write_bytes(encoded.tobytes())
    return img


def test_lru_capacity_limit_and_eviction_order(tmp_path: Path) -> None:
    """Kapasite aşıldığında en eski (LRU) karelerin çıkarıldığını, erişilen karenin korunduğunu doğrular."""
    assert DEFAULT_CACHE_CAPACITY == 32
    cache = LRUFrameDecodeCache(capacity=4)
    assert cache.capacity == 4

    paths = []
    for idx in range(6):
        p = tmp_path / f"frame_{idx:02d}.png"
        _write_test_image(p, fill_value=10 * (idx + 1))
        paths.append(p)

    # İlk 4 kareyi yükle (kapasite = 4 doldu)
    for idx in range(4):
        decoded = cache.get_or_decode(paths[idx])
        assert decoded is not None
        assert int(decoded[0, 0, 0]) == 10 * (idx + 1)

    assert len(cache) == 4
    assert cache.evictions == 0

    # frame_00 ve frame_01'e tekrar erişerek onları MRU (en güncel) konumuna taşı
    assert cache.get_or_decode(paths[0]) is not None
    assert cache.get_or_decode(paths[1]) is not None
    assert cache.hits == 2

    # frame_04 ve frame_05 eklendiğinde en eski olan frame_02 ve frame_03 atılmalı
    assert cache.get_or_decode(paths[4]) is not None
    assert cache.get_or_decode(paths[5]) is not None

    assert len(cache) == 4
    assert cache.evictions == 2
    assert paths[0] in cache
    assert paths[1] in cache
    assert paths[4] in cache
    assert paths[5] in cache
    assert paths[2] not in cache
    assert paths[3] not in cache

    # Dinamik yeniden boyutlandırma (resize) kapasiteyi düşürünce LRU elemanları tahliye etmeli
    cache.resize(2)
    assert cache.capacity == 2
    assert len(cache) == 2
    assert cache.evictions == 4
    assert paths[4] in cache
    assert paths[5] in cache


def test_mtime_change_triggers_automatic_invalidation_and_reload(tmp_path: Path) -> None:
    """Diskteki dosyanın mtime değeri değiştiğinde bayat önbelleğin silinip yeniden okunduğunu doğrular."""
    decode_calls: list[str] = []

    def counting_decoder(path: Path | str) -> np.ndarray | None:
        decode_calls.append(str(path))
        return imread_unicode(path)

    cache = LRUFrameDecodeCache(capacity=8, decoder=counting_decoder)
    img_path = tmp_path / "dynamic_frame.png"
    _write_test_image(img_path, fill_value=42)

    # 1. İlk okuma: diskten decode edilir (miss = 1)
    first_img = cache.get_or_decode(img_path)
    assert first_img is not None
    assert int(first_img[0, 0, 0]) == 42
    assert len(decode_calls) == 1
    assert cache.misses == 1
    assert cache.hits == 0

    # 2. İkinci okuma (mtime aynı): önbellekten döner, diskten tekrar decode edilmez (hit = 1)
    second_img = cache.get_or_decode(img_path)
    assert second_img is not None
    assert int(second_img[0, 0, 0]) == 42
    assert len(decode_calls) == 1
    assert cache.hits == 1
    assert cache.stale_invalidations == 0

    # 3. Dosyayı yeni içerikle güncelle ve mtime değerini ileri sar
    old_stat = img_path.stat()
    _write_test_image(img_path, fill_value=215)
    new_mtime = old_stat.st_mtime + 10.0
    os.utime(img_path, (new_mtime, new_mtime))

    # 4. Üçüncü okuma: mtime farkı algılanmalı, bayat kayıt silinip yeni görüntü yüklenmeli
    reloaded_img = cache.get_or_decode(img_path)
    assert reloaded_img is not None
    assert int(reloaded_img[0, 0, 0]) == 215
    assert len(decode_calls) == 2
    assert cache.stale_invalidations == 1
    assert cache.misses == 2

    # 5. Yeni mtime ile tekrar okuma: artık önbellekten isabet (hit) almalı
    cached_again = cache.get_or_decode(img_path)
    assert cached_again is not None
    assert int(cached_again[0, 0, 0]) == 215
    assert len(decode_calls) == 2
    assert cache.hits == 2


def test_deleted_or_replaced_file_invalidates_stale_entry_on_lookup(tmp_path: Path) -> None:
    """Önbelleğe alınmış bir dosya diskten silindiğinde bayat verinin sunulmadığını doğrular."""
    cache = LRUFrameDecodeCache(capacity=8)
    img_path = tmp_path / "transient_frame.png"
    _write_test_image(img_path, fill_value=99)

    loaded = cache.get_or_decode(img_path)
    assert loaded is not None
    assert img_path in cache
    assert len(cache) == 1

    # Dosyayı diskten sil
    img_path.unlink()

    # Önbellek silinen dosyayı otomatik olarak geçersiz kılmalı
    assert cache.get_or_decode(img_path) is None
    assert img_path not in cache
    assert len(cache) == 0
    assert cache.stale_invalidations >= 1


def test_scrubbing_cache_hit_rate_and_single_decode_per_frame(tmp_path: Path) -> None:
    """İleri/geri hızlı kare gezinmesinde (scrubbing) her karenin yalnızca 1 kez decode edildiğini
    ve yüksek önbellek isabet oranına (hit_rate) ulaşıldığını doğrular.
    """
    decode_counts: dict[str, int] = {}

    def instrumented_decoder(path: Path | str) -> np.ndarray | None:
        key = str(Path(path).resolve())
        decode_counts[key] = decode_counts.get(key, 0) + 1
        return imread_unicode(path)

    cache = LRUFrameDecodeCache(capacity=16, decoder=instrumented_decoder)
    frames = []
    for idx in range(12):
        p = tmp_path / f"scrub_{idx:03d}.png"
        _write_test_image(p, fill_value=idx + 1, shape=(64, 64, 3))
        frames.append(p)

    # Komşu ön-yükleme (prefetch_neighbors) ile 5. karenin etrafını ısıt
    prefetched = cache.prefetch_neighbors(frames, current_index=5, radius=2)
    assert prefetched == 5

    # 12 kare üzerinde 5 tam tur ileri-geri gezinme simülasyonu (toplam 120 okuma)
    for _ in range(5):
        for p in frames:
            img = cache.get_or_decode(p)
            assert img is not None
        for p in reversed(frames):
            img = cache.get_or_decode(p)
            assert img is not None

    stats = cache.get_stats()
    # 12 karenin tamamı kapasiteye (16) sığdığı için her dosya diskten tam 1 kez decode edilmeli
    assert len(decode_counts) == 12
    assert all(count == 1 for count in decode_counts.values())
    assert stats["evictions"] == 0
    assert stats["misses"] == 12
    assert stats["hits"] >= 110
    assert stats["hit_rate"] >= 0.90
    assert stats["memory_bytes"] > 0


def test_unicode_turkish_path_and_thread_safe_concurrent_access(tmp_path: Path) -> None:
    """Türkçe karakterli dosya yollarında sorunsuz decode yapıldığını ve çoklu iş parçacığı
    altında kapasite sınırının ve veri bütünlüğünün korunduğunu doğrular.
    """
    unicode_dir = tmp_path / "selçuk_ğüşıöç_önbellek"
    unicode_dir.mkdir(parents=True, exist_ok=True)

    cache = LRUFrameDecodeCache(capacity=8, copy_on_read=True)
    paths: list[Path] = []
    for idx in range(16):
        p = unicode_dir / f"kare_ğüşıöç_{idx:02d}.png"
        _write_test_image(p, fill_value=5 * (idx + 1))
        paths.append(p)

    def worker(worker_id: int) -> bool:
        for step in range(24):
            target_idx = (worker_id * 3 + step) % len(paths)
            img = cache.get_or_decode(paths[target_idx])
            if img is None:
                return False
            expected_val = 5 * (target_idx + 1)
            if int(img[0, 0, 0]) != expected_val:
                return False
            # copy_on_read=True sayesinde dönen dizi üzerinde değişiklik önbelleği bozmamalı
            img[0, 0, 0] = 0
        return True

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, range(4)))

    assert all(results)
    assert len(cache) <= cache.capacity
    stats = cache.get_stats()
    assert stats["hits"] + stats["misses"] == 4 * 24
    assert stats["decode_errors"] == 0
