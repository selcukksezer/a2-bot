# -*- coding: utf-8 -*-
"""Faz 2 — YOLO Fabrikası toplu onay / zehir temizliği async testleri.

Kapsam (fix/yolo-bulk-onay-async):
- async toplu onay tamamlanır + dürüst ilerleme raporlanır (P2-21:
  "55 onaylandı · 60/60 işlendi" — işlenen/onaylanan ayrı sayılır)
  + boş kare atlanır (W46) + el etiketi korumalı kare atlanır + UI
  bloklanmaz (en büyük olay-döngüsü boşluğu < 2 sn).
- zehir temizliği async tamamlanır + ilerleme raporlanır.

Canlı log kanıtı (2026-11-10): toplu onay UI'ı 50,8 sn, zehir temizliği
18,8 sn kilitlemişti. Bu testler o donmanın tekrarını yakalar.

Mevcut hiçbir test dosyası düzenlenmedi; fixture deseni
test_yolo_factory_shortcuts_and_bulk.py + el etiketi koruması
test_yolo_factory_hand_label_protection.py'den alınır.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

import dataset_factory.exemplar_memory as exemplar_memory_mod
from dataset_factory.exemplar_memory import ExemplarMemory
from ui.yolo_factory_dialog import HAND_LABEL_TXT_SOURCE, YOLOFactoryDialog

# Async eşiği (_BULK_ASYNC_MIN_ITEMS = 40): async yol ancak 40 kare
# üstünde devreye girer; testler 60 kare kullanır.
N_ASYNC_ITEMS = 60
N_EMPTY = 3
N_PROTECTED = 2
N_NORMAL = N_ASYNC_ITEMS - N_EMPTY - N_PROTECTED  # 55


@pytest.fixture
def factory_dialog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> YOLOFactoryDialog:
    dlg = YOLOFactoryDialog()
    dlg._auto_confirm_dialogs = True
    dlg._dataset_dir = lambda: tmp_path  # type: ignore[assignment]
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    dlg._models_dir = lambda: models_dir  # type: ignore[assignment]
    (tmp_path / "review" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "previews").mkdir(parents=True, exist_ok=True)
    dlg.bulk_range.setValue(90)
    # CWD bağlı yazımlar (models/stale_manifest.json) tmp altına alınır.
    monkeypatch.chdir(tmp_path)
    assert dlg.wait_for_yolo_load(), "async kuyruk yüklemesi 10 sn'de bitmedi"
    dlg._reload_review_queue()
    return dlg


def _box(confidence: float, class_id: int) -> Dict[str, Any]:
    return {
        "confidence": confidence,
        "class_id": class_id,
        "cx": 0.5,
        "cy": 0.5,
        "w": 0.2,
        "h": 0.2,
    }


def _make_sample(
    tmp_path: Path,
    stem: str,
    weak: List[Dict[str, Any]] | None = None,
    trusted: List[Dict[str, Any]] | None = None,
    *,
    write_label_txt: bool = True,
) -> Dict[str, Any]:
    """Gerçek PNG + review JSON + (isteğe bağlı) labels TXT ile kare."""
    import cv2
    import numpy as np

    json_path = tmp_path / "review" / "train" / f"{stem}.json"
    json_path.write_text("{}", encoding="utf-8")
    img_path = tmp_path / "images" / "train" / f"{stem}.png"
    assert cv2.imwrite(str(img_path), np.full((200, 200, 3), (60, 120, 180), dtype=np.uint8))
    if write_label_txt:
        (tmp_path / "labels" / "train" / f"{stem}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    return {
        "stem": stem,
        "split": "train",
        "weak": list(weak or []),
        "trusted": list(trusted or []),
        "done": None,
        "path": str(json_path),
    }


def _protected_item(tmp_path: Path, stem: str) -> Dict[str, Any]:
    """TXT kökenli el etiketi karesi: labels TXT var, review JSON yok.

    Disk kanıtı (`_stem_is_txt_only_frame`) + bellek kanıtı
    (`hand_label=True` + `source="hand_label_txt"`) birlikte kurulur.
    """
    txt_path = tmp_path / "labels" / "train" / f"{stem}.txt"
    txt_path.write_bytes(b"6 0.500000 0.500000 0.100000 0.200000\n")
    import cv2
    import numpy as np

    img_path = tmp_path / "images" / "train" / f"{stem}.png"
    assert cv2.imwrite(str(img_path), np.full((200, 200, 3), (60, 120, 180), dtype=np.uint8))
    return {
        "stem": stem,
        "split": "train",
        "weak": [],
        "trusted": [
            {
                "class_id": 6,
                "class_name": "edana_crest",
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.1,
                "h": 0.2,
                "confidence": 1.0,
                "hand_label": True,
                "trusted": True,
                "source": HAND_LABEL_TXT_SOURCE,
            }
        ],
        "done": None,
        "path": str(tmp_path / "review" / "train" / f"{stem}.json"),
    }


def _record_loading_details(dlg: YOLOFactoryDialog) -> List[str]:
    """Overlay detail yazılarını kaydet (ilerleme sayacı kanıtı)."""
    seen: List[str] = []
    orig = dlg._set_yolo_loading_detail

    def _rec(text: str) -> None:
        seen.append(str(text or ""))
        orig(text)

    dlg._set_yolo_loading_detail = _rec  # type: ignore[method-assign]
    return seen


def _max_event_loop_gap(dlg: YOLOFactoryDialog) -> Dict[str, float]:
    """50 ms'lik QTimer ile olay döngüsü boşluğunu ölç (UI-WATCHDOG deseni)."""
    state = {"last": time.monotonic(), "max_gap": 0.0}

    def _tick() -> None:
        now = time.monotonic()
        state["max_gap"] = max(state["max_gap"], now - state["last"])
        state["last"] = now

    timer = QTimer()
    timer.timeout.connect(_tick)
    timer.start(50)
    state["timer"] = timer  # type: ignore[assignment]
    return state


# ============================================================================
# 1. Async toplu onay
# ============================================================================


def test_bulk_approve_async_completes_with_progress_and_skips(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    """60 kare (>40 eşik) async onaylanır: ilerleme sayacı, W46 boş atlama,
    el etiketi koruması, UI bloklanmaması birlikte doğrulanır."""
    dlg = factory_dialog
    items: List[Dict[str, Any]] = []
    for i in range(N_NORMAL):
        items.append(_make_sample(tmp_path, f"async_ok_{i:03d}", weak=[_box(0.95, 1)]))
    for i in range(N_EMPTY):
        items.append(_make_sample(tmp_path, f"async_empty_{i:03d}", weak=[], trusted=[]))
    for i in range(N_PROTECTED):
        items.append(_protected_item(tmp_path, f"async_prot_{i:03d}"))

    dlg._review_items = items
    dlg.set_queue_filter_mode("all")
    assert len(dlg._filtered_review_indices()) == N_ASYNC_ITEMS

    seen = _record_loading_details(dlg)
    gap = _max_event_loop_gap(dlg)
    try:
        dlg._bulk_approve()
        # Async yol devrede: çağrı hemen döner, overlay görünür.
        assert dlg._bulk_approve_running is True
        assert not dlg._yolo_loading_overlay.isHidden()
        assert dlg.wait_for_bulk_approve(timeout_ms=60000), (
            "async toplu onay 60 sn'de bitmedi"
        )
    finally:
        gap["timer"].stop()

    assert dlg._bulk_approve_running is False
    assert dlg._yolo_loading_overlay.isHidden()

    # İlerleme sayacı tek yönlü, tam sayıda bitti. P2-21 (F043-K2):
    # overlay ONAYLANANI sayar — "Toplu onay: 60/60" yalanı bitti
    # (60 işlenen, onaylanan 55: 2 korumalı + 3 boş atlandı); son
    # satır "55 onaylandı · 60/60 işlendi"dir.
    progress_lines = [s for s in seen if s.startswith("Toplu onay:")]
    assert progress_lines, "hiç ilerleme sayacı yazılmadı"
    assert progress_lines[0] == "Toplu onay: 0/60"
    assert progress_lines[-1] == "Toplu onay: 55 onaylandı · 60/60 işlendi"

    def _done(line: str) -> int:
        # İşlenen sayı: P2-21 satırında "·" sonrasındaki kesrin payı
        # ("55 onaylandı · 60/60 işlendi" -> 60); başlangıç satırı
        # "Toplu onay: 0/60" düz kesirdir.
        tail = line.split("·")[-1] if "·" in line else line.split(":", 1)[1]
        return int(tail.strip().split("/")[0])

    nums = [_done(s) for s in progress_lines]
    assert nums == sorted(nums), "ilerleme sayacı geriye gitti"

    # 55 normal kare onaylanıp kuyruktan çıktı; 3 boş + 2 korumalı kaldı.
    remaining = {it["stem"]: it for it in dlg._review_items}
    assert len(remaining) == N_EMPTY + N_PROTECTED
    for i in range(N_EMPTY):
        stem = f"async_empty_{i:03d}"
        assert stem in remaining
        assert remaining[stem].get("done") is None
        # W46: boş kare diske onaylı yazılmadı.
        assert "ok" not in (tmp_path / "review" / "train" / f"{stem}.json").read_text(
            encoding="utf-8"
        )
    for i in range(N_PROTECTED):
        stem = f"async_prot_{i:03d}"
        assert stem in remaining
        assert remaining[stem].get("done") is None
        assert remaining[stem].get("human_approved") is not True
        # Korumalı kare: TXT bayt-bayt duruyor, review JSON oluşmadı.
        assert (tmp_path / "labels" / "train" / f"{stem}.txt").read_bytes() == (
            b"6 0.500000 0.500000 0.100000 0.200000\n"
        )
        assert not (tmp_path / "review" / "train" / f"{stem}.json").exists()

    log_text = dlg.training_log_box.toPlainText()
    assert "Filtredeki 55 kare onaylandı" in log_text
    assert "[W46] 3 boş kare toplu onaya girmedi" in log_text
    assert "2 el etiketi korumalı kare toplu onaya girmedi" in log_text

    # UI bloklanmadı: en büyük olay-döngüsü boşluğu 2 sn altı
    # (canlı logdaki 50,8 sn'lik kilitlenme tekrarı yakalanır).
    assert gap["max_gap"] < 2.0, f"UI boşluğu çok büyük: {gap['max_gap']:.2f} sn"


def test_bulk_approve_sync_path_still_w46_skips_empty(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    """Küçük toplu onay (<=40, senkron yol) da boş kareyi onaylamaz.

    W46 düzeltmesi çekirdek fonksiyondadır — senkron yolda da `done`
    yazılmaz, yanıltıcı `[HATA]` logu çıkmaz.
    """
    dlg = factory_dialog
    empty = _make_sample(tmp_path, "sync_empty_001", weak=[], trusted=[])
    ok_item = _make_sample(tmp_path, "sync_ok_001", weak=[_box(0.95, 1)])
    dlg._review_items = [empty, ok_item]
    dlg.set_queue_filter_mode("empty")
    assert len(dlg._filtered_review_indices()) == 1

    dlg._bulk_approve()

    by_stem = {it["stem"]: it for it in dlg._review_items}
    assert by_stem["sync_empty_001"].get("done") is None
    assert "sync_empty_001" in by_stem  # kuyrukta kaldı
    assert "ok" not in (tmp_path / "review" / "train" / "sync_empty_001.json").read_text(
        encoding="utf-8"
    )
    assert "[HATA] #sync_empty_001" not in dlg.training_log_box.toPlainText()
    assert "[W46] 1 boş kare toplu onaya girmedi" in dlg.training_log_box.toPlainText()


# ============================================================================
# 2. Async zehir temizliği
# ============================================================================


def test_purge_poisoned_labels_async_completes_with_progress(
    factory_dialog: YOLOFactoryDialog,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zehir temizliği worker'da biter: taban-altı kutular silinir, emin
    kutu korunur, ilerleme sayacı raporlanır, UI yanıt verir kalır."""
    dlg = factory_dialog
    # ExemplarMemory üretimin learned_exemplars.json'una yazmasın.
    monkeypatch.setattr(
        exemplar_memory_mod,
        "ExemplarMemory",
        lambda *a, **k: ExemplarMemory(json_path=tmp_path / "exemplars.json"),
    )
    import cv2
    import numpy as np

    rng = np.random.default_rng(7)
    n_files = 6
    for i in range(n_files):
        stem = f"purge_{i:03d}"
        img_path = tmp_path / "images" / "train" / f"{stem}.png"
        # Düz zemin değil (rastgele gürültü): Katman 2 düz-zemin silmesi yok.
        assert cv2.imwrite(
            str(img_path), rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        )
        payload = {
            "stem": stem,
            "split": "train",
            # Taban-altı zehir (0.30 < eşik 0.90) silinir; emin kutu kalır.
            "weak": [_box(0.30, 1)],
            "trusted": [_box(0.95, 3)],
        }
        (tmp_path / "review" / "train" / f"{stem}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        (tmp_path / "labels" / "train" / f"{stem}.txt").write_text(
            "1 0.5 0.5 0.2 0.2\n3 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )

    dlg._review_items = []
    dlg._reload_review_queue()
    assert len(dlg._review_items) == n_files

    seen = _record_loading_details(dlg)
    gap = _max_event_loop_gap(dlg)
    try:
        dlg._purge_poisoned_labels()
        assert dlg._purge_running is True
        assert not dlg._yolo_loading_overlay.isHidden()
        assert dlg.wait_for_bulk_purge(timeout_ms=120000), (
            "async zehir temizliği 120 sn'de bitmedi"
        )
    finally:
        gap["timer"].stop()

    assert dlg._purge_running is False
    assert dlg._yolo_loading_overlay.isHidden()

    # İlerleme sayacı kümülatif (train+val toplamı), sonda tam sayıda.
    progress_lines = [s for s in seen if s.startswith("Zehir temizliği:")]
    assert progress_lines, "hiç temizlik ilerlemesi yazılmadı"
    assert progress_lines[-1] == f"Zehir temizliği: {n_files}/{n_files}"
    # Başlangıçtaki "Zehir temizliği: taranıyor…" satırı sayı içermez — parse'tan atla.
    nums = [int(s.split(":")[1].strip().split("/")[0]) for s in progress_lines
            if "taranıyor" not in s]
    assert nums == sorted(nums), "temizlik ilerlemesi geriye gitti"

    # Taban-altı 6 kutu silindi, 6 emin kutu diskte duruyor.
    log_text = dlg.training_log_box.toPlainText()
    assert "[TEMİZ YZ]" in log_text
    assert "Silinen=6" in log_text
    for i in range(n_files):
        stem = f"purge_{i:03d}"
        after = json.loads(
            (tmp_path / "review" / "train" / f"{stem}.json").read_text(encoding="utf-8")
        )
        assert len(after.get("weak", [])) == 0
        assert len(after.get("trusted", [])) == 1

    assert gap["max_gap"] < 2.0, f"UI boşluğu çok büyük: {gap['max_gap']:.2f} sn"
