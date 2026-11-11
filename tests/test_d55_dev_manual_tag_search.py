# -*- coding: utf-8 -*-
"""tests/test_d55_dev_manual_tag_search.py
==========================================
DEV slice D55 tests for `dataset_factory/manual_tag_search.py`:

1. Search speed: per-class folded labels / primary tokens / sorted
   keyword lists are precomputed once at registry build, prefix scans use
   bisect, and the prefix/multi-token tiers are skipped once the score
   already reaches their ceiling. A keystroke search must stay well under
   the ~15 ms the old per-search re-folding cost.
2. Ranking preservation: the precompute/bisect rewrite must not change
   any ranking except the intended class-26 improvements (verified against
   a 192-query before/after snapshot; only 9 queries changed, all 26-related).
3. Class 26 (player_character) searchability: the taxonomy aliases
   ("hero", "avatar", "ben", "self") returned nothing from the UI search
   and "oyuncu" ranked the HP/MP bars above the character itself.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_factory.class_taxonomy import CANONICAL_NAMES
from dataset_factory.manual_tag_search import (
    SearchableTagRegistry,
    _has_prefix,
    get_tag_registry,
    search_label_classes,
)


def test_d55_score_fields_precomputed_for_all_classes() -> None:
    """Every class has cached scoring fields, so _score_entry never re-folds."""
    registry = SearchableTagRegistry()
    assert len(registry._score_fields) == len(CANONICAL_NAMES) == 29
    fields = registry._score_fields[0]
    assert fields["cname_ascii"] == "herb_aria"
    assert fields["short_ascii"] == "aria"
    assert fields["label_en_ascii"] == "aria"
    assert "aria" in fields["primary_tokens"]
    assert fields["sorted_kw_ascii"] == sorted(fields["sorted_kw_ascii"])
    assert fields["sorted_tokens"] == sorted(fields["sorted_tokens"])
    assert fields["super_fold"] == "3d_material"
    assert registry._score_fields[26]["super_fold"] == "player"


def test_d55_has_prefix_matches_linear_startswith() -> None:
    """Bisect prefix check agrees with any(startswith) on sorted input."""
    terms = ["apple", "apricot", "banana", "berry"]
    assert _has_prefix(terms, "ap") is True
    assert _has_prefix(terms, "app") is True
    assert _has_prefix(terms, "ba") is True
    assert _has_prefix(terms, "c") is False
    assert _has_prefix(terms, "berry") is True
    assert _has_prefix(terms, "berryx") is False
    assert _has_prefix([], "ap") is False
    # Empty prefix matches any non-empty list, same as str.startswith("").
    assert _has_prefix(terms, "") is True


def test_d55_canonical_rankings_unchanged() -> None:
    """The 9 documented queries and the prefix/multi-token tiers rank as before."""
    expected_top = {
        "aria": (0, "herb_aria"),
        "orichalcum": (3, "ore_orichalcum"),
        "pot": (22, "hp_potion"),
        "iksir": (22, "hp_potion"),
        "scroll": (24, "buff_scroll"),
        "parşömen": (24, "buff_scroll"),
        "canavar": (10, "monster_3d"),
        "filiz": (9, "minimap_sprout"),
        # "skill" ties at 850 for all four skill classes; the shorter
        # canonical name (skill_bar, 19) wins the tie-break.
        "skill": (19, "skill_bar"),
    }
    for query, (exp_id, exp_name) in expected_top.items():
        res = search_label_classes(query)
        assert res, query
        assert (res[0]["class_id"], res[0]["class_name"]) == (exp_id, exp_name), query
    assert {(r["class_id"], r["class_name"]) for r in search_label_classes("skill")} >= {
        (13, "skill_ui_prompt"), (19, "skill_bar"),
        (20, "skill_slot_ready"), (21, "skill_slot_cooldown"),
    }

    # Prefix tier (bisect path): "orich" still prefixes orichalcum at 720.
    assert search_label_classes("orich")[0]["class_id"] == 3
    assert search_label_classes("orich")[0]["score"] == 720.0
    # Multi-token AND tier: "kırmızı kristal" -> blood crystal, "canavar noktası" -> minimap dot.
    assert search_label_classes("kırmızı kristal")[0]["class_id"] == 12
    assert search_label_classes("canavar noktası")[0]["class_id"] == 11
    assert search_label_classes("yol zemin")[0]["class_id"] == 15
    # Numeric id 27 is now canonical (other_ore); out-of-range still empty.
    assert search_label_classes("zzzqqq") == []
    ore27 = search_label_classes("27")
    assert ore27 and ore27[0]["class_id"] == 27
    assert search_label_classes("29") == []
    assert search_label_classes("999") == []


def test_d55_player_character_aliases_found_and_ranked_first() -> None:
    """Taxonomy aliases for class 26 now resolve in the UI search itself."""
    for query in ("hero", "avatar", "ben", "self", "kendi karakterim"):
        res = search_label_classes(query)
        assert res, query
        assert res[0]["class_id"] == 26, query
        assert res[0]["class_name"] == "player_character", query

    # "oyuncu" used to rank the HP bar (16) and MP bar (17) above the
    # character (all tied at 820); the exact keyword now puts 26 first.
    oyuncu = search_label_classes("oyuncu")
    assert oyuncu[0]["class_id"] == 26
    assert oyuncu[0]["score"] == 850.0
    assert {r["class_id"] for r in oyuncu[:3]} == {16, 17, 26}

    # "karakter" was a 720 prefix hit; it is now an exact keyword at 850.
    karakter = search_label_classes("karakter")
    assert karakter[0]["class_id"] == 26
    assert karakter[0]["score"] == 850.0

    # Registry entry carries the new aliases; class id mapping untouched.
    entry = get_tag_registry().get_by_id(26)
    assert entry["class_id"] == 26 and entry["class_name"] == "player_character"
    assert "hero" in entry["aliases"] and "oyuncu" in entry["aliases"]


def test_d55_player_category_label_and_filter() -> None:
    """Super-category 'player' now has a Turkish display label and filter."""
    entry = get_tag_registry().get_by_id(26)
    assert entry["super_category"] == "player"
    assert entry["category_tr"] == "Oyuncu Karakteri"
    # P1-8 (K-21): boş sorgu hiçbir şeyle eşleşmez — kategori filtresi
    # bile boş sorguyu listelemeye döndürmez; katalog listeleme
    # get_all_classes()'ın işidir (dataset_factory/manual_tag_search.py:
    # SearchableTagRegistry.search, boş sorgu -> []).
    assert search_label_classes("", category="player") == []
    assert search_label_classes("", category="oyuncu") == []
    # Kategori filtresi gerçek sorguda çalışır: 'oyuncu' yalnızca
    # player üst-kategorisindeki sınıfı (26) döndürür.
    assert [r["class_id"] for r in search_label_classes("oyuncu", category="player")] == [26]
    assert [r["class_id"] for r in search_label_classes("oyuncu", category="oyuncu")] == [26]


def test_d55_search_latency() -> None:
    """200 mixed keystroke searches complete well under the old ~3.2 s.

    Before the change a single search cost ~15-16 ms (200 x aria/canavar/
    skill measured 3.0-3.2 s); after precompute + bisect the same 200 cost
    ~0.12-0.17 s. The 1.5 s bound leaves >2x headroom over the old time
    and ~10x over the new time, so it locks the regression without
    flaking on a loaded CI runner.
    """
    registry = SearchableTagRegistry()
    queries = [
        "aria", "orichalcum", "pot", "iksir", "scroll", "parşömen",
        "canavar", "skill", "filiz", "engel", "yol", "hero", "oyuncu",
        "maden", "kristal", "bitki", "kırmızı kristal", "canavar noktası",
        "mana", "hp",
    ]
    start = time.perf_counter()
    for _ in range(10):
        for q in queries:
            registry.search(q)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.5, f"200 searches took {elapsed:.3f}s"
