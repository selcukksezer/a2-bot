# -*- coding: utf-8 -*-
"""
tests/test_p2_25_stem_sanitize.py

P2-25 (bölüm 8) hedefli testleri — stem sanitize + etiket yolu koruması:

- Review JSON `stem` alanı dosya adı olarak güvenli hale getirilir:
  `/`, `\\`, `..`, NUL ve boş değer kabul edilmez; geçersiz stem'de
  review dosyasının kendi stem'i (fallback) kullanılır; ikisi de
  geçersizse ValueError.
- `_label_txt_path` çözümlenmiş yolun labels dizini altında kaldığını
  doğrular (sembolik bağ üzerinden dışarı çıkma engellenir).
- `config/.vault_secret` .gitignore'dadır (P2-25: vault secret
  repoya girmez).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_factory.label_trust import _label_txt_path, _sanitize_stem


# ---------------------------------------------------------------------------
# _sanitize_stem
# ---------------------------------------------------------------------------

def test_sanitize_stem_plain_ok():
    assert _sanitize_stem("frame_000123", "fallback") == "frame_000123"


def test_sanitize_stem_traversal_uses_fallback():
    """`../../` içeren stem reddedilir, review dosyasının stem'i kullanılır."""
    assert _sanitize_stem("../../etc/passwd", "frame_0001") == "frame_0001"
    assert _sanitize_stem("..\\..\\windows", "frame_0002") == "frame_0002"
    assert _sanitize_stem("sub/frame", "frame_0003") == "frame_0003"
    assert _sanitize_stem("..", "frame_0004") == "frame_0004"
    assert _sanitize_stem(".", "frame_0005") == "frame_0005"


def test_sanitize_stem_empty_or_none_uses_fallback():
    assert _sanitize_stem("", "frame_0010") == "frame_0010"
    assert _sanitize_stem(None, "frame_0011") == "frame_0011"
    assert _sanitize_stem("   ", "frame_0012") == "frame_0012"


def test_sanitize_stem_nul_rejected():
    assert _sanitize_stem("frame\x00evil", "frame_0020") == "frame_0020"


def test_sanitize_stem_both_invalid_raises():
    """Stem de fallback de geçersizse dosya yazılmaz — ValueError."""
    with pytest.raises(ValueError):
        _sanitize_stem("../../x", "../y")
    with pytest.raises(ValueError):
        _sanitize_stem("", "")
    with pytest.raises(ValueError):
        _sanitize_stem(None, None)


def test_sanitize_stem_strips_whitespace():
    assert _sanitize_stem("  frame_0030  ", "fallback") == "frame_0030"


# ---------------------------------------------------------------------------
# _label_txt_path
# ---------------------------------------------------------------------------

def test_label_txt_path_inside_labels_dir(tmp_path: Path):
    labels_dir = tmp_path / "labels" / "train"
    labels_dir.mkdir(parents=True)
    target = _label_txt_path(labels_dir, "frame_0001")
    assert target == (labels_dir / "frame_0001.txt").resolve()
    assert target.is_relative_to(labels_dir.resolve())


def test_label_txt_path_symlink_escape_blocked(tmp_path: Path):
    """labels_dir içindeki sembolik bağ dışarı çıkarsa yol reddedilir."""
    outside = tmp_path / "outside"
    outside.mkdir()
    labels_dir = tmp_path / "labels" / "train"
    labels_dir.mkdir(parents=True)
    link = labels_dir / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("sembolik bağ kurulamadı")
    # `escape` bir dizin adı gibi davranır ama resolve() dışarı çıkar.
    with pytest.raises(ValueError):
        _label_txt_path(labels_dir, "escape/frame")


def test_label_txt_path_dotdot_stem_blocked(tmp_path: Path):
    labels_dir = tmp_path / "labels" / "train"
    labels_dir.mkdir(parents=True)
    with pytest.raises(ValueError):
        _label_txt_path(labels_dir, "../frame")


# ---------------------------------------------------------------------------
# .gitignore — vault secret
# ---------------------------------------------------------------------------

def test_vault_secret_is_gitignored():
    """`config/.vault_secret` .gitignore'da olmalı (P2-25)."""
    gitignore = _ROOT / ".gitignore"
    assert gitignore.is_file(), ".gitignore bulunamadı"
    lines = [ln.strip() for ln in gitignore.read_text(encoding="utf-8").splitlines()]
    assert "config/.vault_secret" in lines
