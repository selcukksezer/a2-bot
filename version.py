# -*- coding: utf-8 -*-
"""Gatherion — tek sürüm kaynağı (P3-1, K-03).

Uygulama sürümü ("2.0.0", semver) yalnızca burada tanımlanır. Tüm
``version_info`` fonksiyonları, ayar dosyası varsayılanı
(``config.schema.DEFAULT_VERSION``) ve UI sürüm etiketleri bu sabite
delege eder; üretim kodunun hiçbir yerinde "2.0.0" / "v2.0" literali
tutulmaz.

Eski yazım "v2.0" yalnızca geriye dönük uyumluluk için tanınır:
``normalize_version()`` onu kanonik "2.0.0"a çevirir, böylece eski
``settings.json`` dosyaları sürüm uyumsuzluğu uyarısı vermeden yüklenir.

Bu modül hiçbir şey içe aktarmaz — döngüsel import riski yoktur.
"""

__version__ = "2.0.0"

# Eski sürüm yazımları -> kanonik yazım (yalnızca yükleme anında normalizasyon).
_LEGACY_VERSION_ALIASES = {
    "v2.0": __version__,
    "2.0": __version__,
}


def normalize_version(value) -> str:
    """Sürüm dizesini kanonik yazıma çevirir ("v2.0" -> "2.0.0").

    Bilinmeyen sürümler olduğu gibi döner (örn. eski "v1.6.15" dosyası
    sürüm uyumsuzluğu bayrağını hâlâ tetikler); boş değer kanoniğe döner.
    """
    text = str(value or "").strip()
    if not text:
        return __version__
    if text in _LEGACY_VERSION_ALIASES:
        return _LEGACY_VERSION_ALIASES[text]
    return text
