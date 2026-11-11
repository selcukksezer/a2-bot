# -*- coding: utf-8 -*-
"""Tek exemplar-benzerlik eşiği — tek tanım (P3-3).

`core/vision.py`, `core/skill_bar_reader.py`, `core/mana_phase_combat.py` ve
`core/neural_audio_perception.py` eskiden `_load_exemplar_match_threshold`
fonksiyonunun dört ayrı kopyasını taşıyordu; dördü de bu yardımcıya delege
eder. Davranış aynıdır: `config/settings.json` içinde
`exemplar_match_threshold` varsa o okunur, yoksa 0.68 varsayılır; değer
[0.50, 0.95] aralığına kıskaçlanır. Eskiden 0.65 / 0.68 / 0.70 / 0.72 /
0.75 gibi dağınık değerler kullanılıyordu; hiçbiri ayarda değildi.

Bu modül proje-içi başka hiçbir modüle bağımlı değildir (yalnızca
`json`/`os`), bu yüzden dört çağrıcı da döngüsel import riski olmadan
içe aktarabilir.

Aynı modül `is_negative_exemplar_name`in de tek kaynağıdır (P3-3 v19):
`core/vision.py`, `core/skill_bar_reader.py`, `core/mana_phase_combat.py`
ve `core/neural_audio_perception.py` eskiden bu fonksiyonun dört ayrı
kopyasını taşıyordu; dördü de birebir aynı gövdeydi.
"""

from __future__ import annotations

import json
import os
from typing import Optional

DEFAULT_EXEMPLAR_MATCH_THRESHOLD: float = 0.68
MIN_EXEMPLAR_MATCH_THRESHOLD: float = 0.50
MAX_EXEMPLAR_MATCH_THRESHOLD: float = 0.95


def load_exemplar_match_threshold(
    default: float = DEFAULT_EXEMPLAR_MATCH_THRESHOLD,
) -> float:
    """Exemplar-benzerlik eşiğini ayardan oku, yoksa `default` döndür.

    Okuma başarısız olursa (dosya yok, bozuk JSON, sayı değil) `default`
    döner; okunan değer `[0.50, 0.95]` aralığına kıskaçlanır.
    """
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(
            os.path.join(root, "config", "settings.json"), "r", encoding="utf-8"
        ) as f:
            data = json.load(f)
        val = float(data.get("exemplar_match_threshold", default))
        return min(
            MAX_EXEMPLAR_MATCH_THRESHOLD,
            max(MIN_EXEMPLAR_MATCH_THRESHOLD, val),
        )
    except Exception:
        return float(default)


def is_negative_exemplar_name(class_name: Optional[str]) -> bool:
    """Pozitif listeye sızmış negatif kayıtları yakalar.

    `dataset/learned_exemplars.json` içinde 7 adet (10,
    'monster_3d_negative') kaydı POZİTİF listede duruyor. ExemplarMemory
    bunları `classify_patch` ile pozitif eşleşme olarak döndürebiliyor
    (class_id 10 = monster_3d), yani reddedilmiş bir canavar karesi
    "canavar kanıtı" sayılıyordu. Kalıcı temizlik K grubunda; okuyucu
    tarafı savunması: adı `_negative` ile biten eşleşme asla pozitif
    kanıt sayılmaz, negatif kanıt sayılır.
    """
    name = str(class_name or "").strip().lower()
    return name.endswith("_negative") or name.endswith("-negative") or "negative" in name
