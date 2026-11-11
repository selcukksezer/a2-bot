# -*- coding: utf-8 -*-
"""Stem Material Validator for Dataset Factory (`dataset_factory/stem_material_validator.py`).

Compares material hints embedded in image/label filenames or stems
(e.g. `boot_*_aria_*`, `mela`, `targena`, `orichalcum`, `obsidian`, `wood`)
against the predicted class of a bounding box. When a conflict is detected,
demotes the box confidence to `<= 0.45` and sets:
    `trusted = False`
    `weak = True`
    `demoted_reason = "stem_material_conflict"`

Wiring (F-2001): exported from `dataset_factory/__init__.py` and used as
the material-conflict detector by `tools/recalibrate_dataset_trust.py`
Rule 4 (`is_conflicting_material` delegates to
`is_stem_material_conflict`); the inline stem->class table that used to
live in the recalibration tool was removed in favour of this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union


MAX_CONFLICT_CONFIDENCE: float = 0.45
# B-09: olculmemis guvene varsayilan yazilmaz — DEFAULT_FALLBACK_CONFIDENCE
# (0.40) kaldirildi. _extract_confidence guven alani yoksa None doner;
# eksiklik StemValidationResult.confidence_missing ile acikca isaretlenir.
DEMOTED_REASON_STEM_MATERIAL_CONFLICT: str = "stem_material_conflict"

# Canonical material family keys -> all synonym tokens in stems or class names
MATERIAL_FAMILY_SYNONYMS: Dict[str, FrozenSet[str]] = {
    "aria": frozenset({"aria", "herb_aria", "fresh_aria", "taze_aria", "pella", "valerian", "amilis"}),
    "mela": frozenset({"mela", "herb_mela", "fruit_mela", "fresh_mela", "taze_mela", "berry"}),
    "targena": frozenset({
        "targena",
        "herb_targena",
        "fresh_targena",
        "azpha",
        "herb_azpha",
        "calendula",
        "herb_calendula",
        "lura",
        "gravi",
        "miana",
        "rina",
        "mandragora",
        "atri",
        "kalgelly",
        "lolo",
        "marle",
        "caprauna",
    }),
    "orichalcum": frozenset({
        "orichalcum",
        "ore_orichalcum",
        "orichalcum_ore",
        "ore_iron_silver",
        "ore_iron_copper",
        "iron",
        "titanium",
        "adamantium",
        "mithril",
        "copper",
        "silver",
    }),
    "crystal": frozenset({
        "crystal",
        "gem_crystal",
        "gem",
        "sapphire",
        "gem_sapphire",
        "ruby",
        "gem_ruby",
        "diamond",
        "gem_diamond",
        "emerald",
        "amethyst",
        "garnet",
        "topaz",
        "opal",
        "aquamarine",
        "moonstone",
        "lazuli",
        "odyle",
        "aether",
        "lunime",
    }),
    "wood": frozenset({
        "wood",
        "tree",
        "log",
        "tree_log",
        "wood_tree",
        "log_timber",
        "log_yggdrasil",
        "timber",
        "yggdrasil",
        "asvata",
        "acacia",
        "cypress",
        "elm",
        "oak",
        "pine",
        "maple",
        "cedar",
        "karpaso",
        "nepenthes",
        "kereste",
    }),
    "edana": frozenset({"edana", "herb_edana"}),
    "lumina": frozenset({"lumina", "herb_lumina"}),
    "obsidian": frozenset({"obsidian", "ore_obsidian", "obsidian_ore"}),
    "blood": frozenset({"blood", "crystal_blood", "blood_crystal"}),
}

# Reverse lookup: token -> canonical material family
TOKEN_TO_MATERIAL_FAMILY: Dict[str, str] = {}
for _family, _tokens in MATERIAL_FAMILY_SYNONYMS.items():
    for _tok in _tokens:
        TOKEN_TO_MATERIAL_FAMILY[_tok] = _family
        for _sub in _tok.split("_"):
            if _sub and _sub not in ("herb", "ore", "gem", "fresh", "taze", "fruit"):
                TOKEN_TO_MATERIAL_FAMILY.setdefault(_sub, _family)

# --- Precompiled patterns (D59) -------------------------------------------------
# The hot paths below used to call re.findall / re.split / re.search on every
# invocation. The patterns are built once here from the module constants above
# (nothing else mutates them), so per-call work is dict lookups + one scan.
_STEM_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".json", ".txt")

_BOOT_MATERIAL_RE = re.compile(
    r"boot(?:_[a-z0-9]+)*?_(aria|mela|targena|orichalcum|obsidian|wood|tree|log|crystal|gem|edana|lumina|blood)(?:_|$)"
)
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")

# Concatenated-stem substring fallback (same (token, family) pairs and order
# as the old per-token re.search loop, longest token first so a compound
# token wins over its prefix, e.g. "calendula" over "aria"-style overlaps).
_STEM_SUBSTRING_TOKENS: Tuple[Tuple[str, str], ...] = (
    ("orichalcum", "orichalcum"),
    ("obsidian", "obsidian"),
    ("targena", "targena"),
    ("calendula", "targena"),
    ("yggdrasil", "wood"),
    ("asvata", "wood"),
    ("aria", "aria"),
    ("mela", "mela"),
    ("wood", "wood"),
    ("edana", "edana"),
    ("lumina", "lumina"),
)
_STEM_SUBSTRING_FAMILY: Dict[str, str] = dict(_STEM_SUBSTRING_TOKENS)
_STEM_SUBSTRING_RE = re.compile(
    r"(?<![a-z])(?:"
    + "|".join(re.escape(tok) for tok, _ in sorted(_STEM_SUBSTRING_TOKENS, key=lambda kv: -len(kv[0])))
    + r")(?![a-z])"
)

# Box-class substring fallback: one compiled alternation over every
# TOKEN_TO_MATERIAL_FAMILY key with len >= 4 (exactly the set the old loop
# scanned, ~100 re.search calls per invocation). Longest first so compound
# tokens ("crystal_blood", "gem_crystal") win over their parts. Every
# compound key's parts map to the same family, except crystal_blood /
# blood_crystal, whose "crystal" part is guarded by has_blood_crystal at the
# call site exactly as before -- so a single left-to-right finditer pass
# returns the same family set as the old "check every token" loop.
_BOX_SUBSTRING_KEYS: Tuple[str, ...] = tuple(
    sorted(
        (tok for tok in TOKEN_TO_MATERIAL_FAMILY if len(tok) >= 4),
        key=lambda tok: (-len(tok), tok),
    )
)
_BOX_SUBSTRING_RE = re.compile(
    r"(?<![a-z])(?:" + "|".join(re.escape(tok) for tok in _BOX_SUBSTRING_KEYS) + r")(?![a-z])"
)

# Canonical 22/25-class IDs -> canonical material family
CLASS_ID_TO_MATERIAL_FAMILY: Dict[int, str] = {
    0: "aria",
    1: "mela",
    2: "targena",
    3: "orichalcum",
    4: "crystal",
    5: "wood",
    6: "edana",
    7: "lumina",
    8: "obsidian",
    12: "blood",
}

# UI / gather prompt class tokens that share legacy IDs 6, 7, 8 in 12-class layouts
_NON_MATERIAL_UI_CLASS_TOKENS: FrozenSet[str] = frozenset({
    "world_gather_crest",
    "f_gather_prompt",
    "e_floating_icon",
    "e_gather_icon",
    "skill_ui_prompt",
    "minimap_sprout",
    "minimap_monster_dot",
    "minimap_monster_negative",
})

_3D_STEM_INDICATORS: Tuple[str, ...] = (
    "boot",
    "3d",
    "crest",
    "gather",
    "ref_",
    "node",
)


@dataclass(frozen=True)
class StemValidationResult:
    """Detailed outcome of validating a single box against a filename/stem material hint."""

    conflict: bool
    stem: str
    stem_materials: Tuple[str, ...]
    box_materials: Tuple[str, ...]
    class_id: int
    class_name: str
    # B-09: guven olculmemisse None'dir (uydurma varsayilan yok).
    original_confidence: Optional[float]
    calibrated_confidence: Optional[float]
    trusted: bool
    weak: bool
    # B-09: kutuda hic guven alani yoksa True — cagiran bu kutunun
    # guvenini olculmemis saymalidir.
    confidence_missing: bool = False
    demoted_reason: Optional[str] = None
    box: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict": self.conflict,
            "stem": self.stem,
            "stem_materials": list(self.stem_materials),
            "box_materials": list(self.box_materials),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "original_confidence": self.original_confidence,
            "calibrated_confidence": self.calibrated_confidence,
            "trusted": self.trusted,
            "weak": self.weak,
            "confidence_missing": self.confidence_missing,
            "demoted_reason": self.demoted_reason,
            "box": dict(self.box),
        }


@lru_cache(maxsize=1024)
def _cached_stem_materials(s_lower: str) -> FrozenSet[str]:
    """Memoized stem-material extraction over an already-normalized stem.

    The alias map and the heuristic fallback depend only on the lowered
    string, so repeated labels -- the same few stems on every frame --
    resolve from cache. Returns a frozenset; callers must copy before
    mutating.
    """
    found: Set[str] = set()

    # Check explicit boot_*_<material>_* pattern first
    for token in _BOOT_MATERIAL_RE.findall(s_lower):
        fam = TOKEN_TO_MATERIAL_FAMILY.get(token)
        if fam:
            found.add(fam)

    # Compound blood crystal check before generic crystal check
    has_blood_crystal = "blood_crystal" in s_lower or "crystal_blood" in s_lower
    if has_blood_crystal:
        found.add("blood")

    # Tokenize stem on non-alphanumeric separators
    tokens = [t for t in _TOKEN_SPLIT_RE.split(s_lower) if t]
    for tok in tokens:
        if tok == "crystal" and has_blood_crystal:
            continue
        fam = TOKEN_TO_MATERIAL_FAMILY.get(tok)
        if fam is not None:
            found.add(fam)

    # Substring fallback for concatenated stems (e.g., `boot01aria`, `orichalcum02`).
    # A bare `in` check matched unrelated words mid-token (e.g. "aria"
    # inside "barista"), so a token must start/end at a non-letter
    # boundary; digits next to it are still allowed (boot01aria).
    if not found:
        for match in _STEM_SUBSTRING_RE.finditer(s_lower):
            found.add(_STEM_SUBSTRING_FAMILY[match.group(0)])

    return frozenset(found)


class StemMaterialValidator:
    """Validates bounding box predictions against filename/stem material hints.

    When a filename/stem contains a known material hint (such as `boot_*_aria_*`,
    `mela`, `targena`, `orichalcum`, `obsidian`, `wood`) and the predicted box class
    conflicts with that material, the box's confidence is demoted to `<= 0.45` with
    `trusted=False`, `weak=True`, and `demoted_reason="stem_material_conflict"`.
    """

    def __init__(
        self,
        max_conflict_confidence: float = MAX_CONFLICT_CONFIDENCE,
        demoted_reason: str = DEMOTED_REASON_STEM_MATERIAL_CONFLICT,
        protect_hand_labels: bool = True,
    ) -> None:
        # B-09: default_fallback_confidence parametresi kaldirildi — olculmemis
        # guvene varsayilan yazan tek tuketici _extract_confidence idi ve o da
        # artik None donuyor; nitelik hicbir yerde okunmuyordu.
        if max_conflict_confidence > MAX_CONFLICT_CONFIDENCE or max_conflict_confidence <= 0.0:
            raise ValueError(
                f"max_conflict_confidence must be in (0.0, {MAX_CONFLICT_CONFIDENCE}], "
                f"got {max_conflict_confidence}"
            )
        self.max_conflict_confidence: float = float(max_conflict_confidence)
        self.demoted_reason: str = str(demoted_reason or DEMOTED_REASON_STEM_MATERIAL_CONFLICT)
        self.protect_hand_labels: bool = bool(protect_hand_labels)

    @staticmethod
    def normalize_stem(stem_or_path: Union[str, Path, None]) -> str:
        """Extract normalized lowercase stem from a filename, path, or stem string.

        Pure-string fast path with ``pathlib.Path(...).name`` parity on POSIX
        (backslash is NOT a separator): drop trailing slashes, keep the part
        after the last ``"/"``, lowercase, strip known extensions repeatedly.
        """
        if stem_or_path is None:
            return ""
        raw = str(stem_or_path).strip()
        if not raw:
            return ""
        name = (
            raw
            if "/" not in raw and "\\" not in raw and raw not in (".", "..")
            else Path(raw).name
        )
        # Fast path above: with no path separator Path(raw).name == raw
        # (except "." / ".."); avoids a Path() build per call —
        # normalize_stem runs 3x per validate_box call.
        lower_name = name.lower()
        # Strip ALL known extensions (e.g. "img.png.json" -> "img"), not
        # just the first one: the old code broke out after a single strip.
        stripped = True
        while stripped:
            stripped = False
            for ext in _STEM_EXTENSIONS:
                if lower_name.endswith(ext):
                    lower_name = lower_name[: -len(ext)]
                    stripped = True
                    break
        return lower_name.strip()

    @classmethod
    def extract_stem_materials(cls, stem_or_path: Union[str, Path, None]) -> Set[str]:
        """Extract canonical material families hinted by the filename/stem.

        Handles patterns like `boot_*_aria_*`, `ref_mela_02`, `targena`,
        `orichalcum`, `obsidian`, `wood`, `herb_edana`, `crystal_blood`, etc.
        Results are memoized on the normalized stem (the same few stems
        recur on every frame); a fresh ``set`` copy is returned so callers
        cannot poison the cache by mutating the result.
        """
        s_lower = cls.normalize_stem(stem_or_path)
        if not s_lower:
            return set()
        return set(_cached_stem_materials(s_lower))

    @classmethod
    def extract_box_materials(
        cls,
        class_name: Optional[str] = None,
        class_id: Optional[int] = None,
    ) -> Set[str]:
        """Extract canonical material families represented by a predicted box class."""
        c_lower = str(class_name or "").strip().lower()
        cid: int = -1
        if class_id is not None and not isinstance(class_id, bool):
            try:
                cid = int(class_id)
            except (TypeError, ValueError):
                cid = -1

        # If class_name explicitly indicates a non-material gather UI icon (legacy 12-class 6/7/8),
        # do not treat it as a 3D material class.
        if c_lower in _NON_MATERIAL_UI_CLASS_TOKENS or any(
            ui_tok in c_lower for ui_tok in ("f_gather", "e_floating", "e_gather", "gather_prompt", "gather_crest")
        ):
            return set()

        found: Set[str] = set()

        if c_lower:
            has_blood_crystal = "blood_crystal" in c_lower or "crystal_blood" in c_lower
            if has_blood_crystal:
                found.add("blood")

            tokens = [t for t in _TOKEN_SPLIT_RE.split(c_lower) if t]
            for tok in tokens:
                if tok == "crystal" and has_blood_crystal:
                    continue
                fam = TOKEN_TO_MATERIAL_FAMILY.get(tok)
                if fam is not None:
                    found.add(fam)

            if not found:
                # Single compiled alternation over the same >=4-char token
                # set the old loop re.search'ed one by one (~100 regex
                # calls per invocation). Longest token first, so compound
                # tokens win over their parts; the "crystal" guard is kept
                # exactly as before.
                for match in _BOX_SUBSTRING_RE.finditer(c_lower):
                    token = match.group(0)
                    if token == "crystal" and has_blood_crystal:
                        continue
                    found.add(TOKEN_TO_MATERIAL_FAMILY[token])

        # Resolve from canonical class_id if class_name didn't already resolve to a conflicting family
        if cid in CLASS_ID_TO_MATERIAL_FAMILY and not found:
            # Guard legacy class_id=12 "world_material" when unnamed vs "crystal_blood"
            if cid == 12 and c_lower == "world_material":
                pass
            else:
                found.add(CLASS_ID_TO_MATERIAL_FAMILY[cid])

        return found

    @classmethod
    def is_invalid_essence_on_material_or_3d_stem(
        cls,
        stem_or_path: Union[str, Path, None],
        class_name: Optional[str] = None,
        class_id: Optional[int] = None,
        stem_materials: Optional[Set[str]] = None,
    ) -> bool:
        """Detect invalid raw `essence_*` or out-of-range class IDs on 3D/material stems."""
        s_lower = cls.normalize_stem(stem_or_path)
        if not s_lower:
            return False

        c_lower = str(class_name or "").strip().lower()
        cid = -1
        if class_id is not None and not isinstance(class_id, bool):
            try:
                cid = int(class_id)
            except (TypeError, ValueError):
                cid = -1

        is_3d_stem = any(k in s_lower for k in _3D_STEM_INDICATORS)
        mats = stem_materials if stem_materials is not None else cls.extract_stem_materials(s_lower)

        if not is_3d_stem and not mats:
            return False

        # Raw essence class or out-of-taxonomy class ID (>= 25) on a
        # 3D/material stem is invalid regardless of whether its material
        # token happens to match the stem's. (The old code computed the
        # match and then returned True on both branches — dead code.)
        if "essence" in c_lower or cid >= 25:
            return True

        return False

    def is_conflict(
        self,
        stem_or_path: Union[str, Path, Dict[str, Any]],
        predicted_class: Union[str, int, Dict[str, Any], None] = None,
        class_id: Optional[int] = None,
    ) -> bool:
        """Return True if the stem's material hint conflicts with the box's predicted class.

        Supports flexible call signatures:
        - `validator.is_conflict("boot_01_aria_node", "herb_mela")`
        - `validator.is_conflict("boot_01_aria_node", 1)`
        - `validator.is_conflict("boot_01_aria_node", {"class_id": 1, "class_name": "herb_mela"})`
        - `validator.is_conflict("boot_01_aria_node", "herb_mela", class_id=1)`
        - `validator.is_conflict({"stem": "boot_01_aria_node", "class_id": 1})`
        """
        stem_str, cname, cid = self._unpack_conflict_args(stem_or_path, predicted_class, class_id)
        if not stem_str:
            return False

        stem_mats = self.extract_stem_materials(stem_str)
        if self.is_invalid_essence_on_material_or_3d_stem(stem_str, cname, cid, stem_mats):
            return True

        box_mats = self.extract_box_materials(cname, cid)
        return self._has_material_conflict(stem_mats, box_mats, cname, cid)

    @classmethod
    def _has_material_conflict(
        cls,
        stem_mats: Set[str],
        box_mats: Set[str],
        cname: str,
        cid: int,
    ) -> bool:
        """True when a stem material hint disagrees with the box class.

        A box whose class resolves to a different material family conflicts.
        A box whose class resolves to NO material family at all (unknown /
        unmapped class) also conflicts when the stem carries a material
        hint: it used to pass silently even though it cannot be verified
        against the hint. Known non-material canonical classes (monster,
        UI prompts, potions, ...) are not material claims and do not
        conflict; neither does a box with no class information at all.
        """
        if not stem_mats:
            return False
        if box_mats:
            return not (stem_mats & box_mats)
        return cls._is_unresolved_class(cname, cid)

    @classmethod
    def _is_unresolved_class(cls, cname: str, cid: int) -> bool:
        """True when the box claims a class that maps to no known family
        and is not a known non-material class."""
        if not cname and cid < 0:
            return False  # no class claim at all; nothing to judge
        if cname in _NON_MATERIAL_UI_CLASS_TOKENS or any(
            ui_tok in cname
            for ui_tok in ("f_gather", "e_floating", "e_gather", "gather_prompt", "gather_crest")
        ):
            return False  # known non-material UI class, not a material claim
        if cid >= 0:
            try:
                from dataset_factory.class_taxonomy import dataset_class_names

                if cid in dataset_class_names():
                    return False  # known canonical class (e.g. monster_3d)
            except Exception:
                if 0 <= cid <= 26:
                    return False
            return True  # class id outside the canonical taxonomy
        return True  # named class that resolves to no material family

    def validate_box(
        self,
        stem_or_box: Union[str, Path, Dict[str, Any]],
        box_or_stem: Union[Dict[str, Any], str, Path, None] = None,
        *,
        in_place: bool = True,
    ) -> Dict[str, Any]:
        """Validate a bounding box dict against the stem material hint and demote on conflict.

        Accepts either `(stem, box)` or `(box, stem)` or `(box,)` (when `box` contains `"stem"`).
        If a conflict is found, sets:
            `box["confidence"] <= 0.45`
            `box["trusted"] = False`
            `box["weak"] = True`
            `box["demoted_reason"] = "stem_material_conflict"`
        Returns the validated (and possibly demoted) box dictionary.
        """
        result = self.evaluate_box(stem_or_box, box_or_stem, in_place=in_place)
        return result.box

    def evaluate_box(
        self,
        stem_or_box: Union[str, Path, Dict[str, Any]],
        box_or_stem: Union[Dict[str, Any], str, Path, None] = None,
        *,
        in_place: bool = True,
    ) -> StemValidationResult:
        """Evaluate a box and return a structured `StemValidationResult`."""
        stem_str, raw_box = self._resolve_stem_and_box(stem_or_box, box_or_stem)
        target_box = raw_box if in_place else dict(raw_box)

        norm_stem = self.normalize_stem(stem_str)
        stem_mats = self.extract_stem_materials(norm_stem)
        return self._evaluate_box_prepared(norm_stem, stem_mats, target_box)

    def _evaluate_box_prepared(
        self,
        norm_stem: str,
        stem_mats: Set[str],
        target_box: Dict[str, Any],
    ) -> StemValidationResult:
        """Evaluate one (already copied, if needed) box against precomputed stem data.

        Batch entry point: `validate_boxes` / `validate_review_payload`
        normalize the stem and extract its materials once instead of once
        per box. Behaviour is identical to `evaluate_box`; only the repeated
        stem work is hoisted. `target_box` is mutated on demotion.
        """
        cid = self._extract_class_id(target_box)
        cname = self._extract_class_name(target_box, cid)
        orig_conf = self._extract_confidence(target_box)
        # B-09: guven yoksa None kalir; asla 0.40 yazilmaz.
        conf_missing = orig_conf is None

        box_mats = self.extract_box_materials(cname, cid)

        if self.protect_hand_labels and self._is_protected_hand_label(target_box):
            return StemValidationResult(
                conflict=False,
                stem=norm_stem,
                stem_materials=tuple(sorted(stem_mats)),
                box_materials=tuple(sorted(box_mats)),
                class_id=cid,
                class_name=cname,
                original_confidence=orig_conf,
                calibrated_confidence=orig_conf,
                trusted=bool(target_box.get("trusted", True)),
                weak=bool(target_box.get("weak", False)),
                confidence_missing=conf_missing,
                demoted_reason=target_box.get("demoted_reason"),
                box=target_box,
            )

        conflict = False
        if self.is_invalid_essence_on_material_or_3d_stem(norm_stem, cname, cid, stem_mats):
            conflict = True
        elif self._has_material_conflict(stem_mats, box_mats, cname, cid):
            conflict = True

        if conflict:
            # Never inflate: a zero/negative confidence stays zero. The old
            # code replaced it with the 0.40 fallback, promoting a
            # zero-confidence box to 0.40 by the demotion itself.
            # B-09: guven olculmemisse dusurulecek deger yoktur — kutuya
            # uydurma guven yazilmaz; yalnizca trusted/weak/demoted_reason
            # isaretlenir ve confidence_missing tasinir.
            if orig_conf is None:
                demoted_conf = None
            else:
                demoted_conf = round(max(0.0, min(orig_conf, self.max_conflict_confidence)), 4)
                target_box["confidence"] = demoted_conf
                if "conf" in target_box:
                    target_box["conf"] = demoted_conf
            target_box["trusted"] = False
            target_box["weak"] = True
            target_box["demoted_reason"] = self.demoted_reason

            return StemValidationResult(
                conflict=True,
                stem=norm_stem,
                stem_materials=tuple(sorted(stem_mats)),
                box_materials=tuple(sorted(box_mats)),
                class_id=cid,
                class_name=cname,
                original_confidence=orig_conf,
                calibrated_confidence=demoted_conf,
                trusted=False,
                weak=True,
                confidence_missing=conf_missing,
                demoted_reason=self.demoted_reason,
                box=target_box,
            )

        return StemValidationResult(
            conflict=False,
            stem=norm_stem,
            stem_materials=tuple(sorted(stem_mats)),
            box_materials=tuple(sorted(box_mats)),
            class_id=cid,
            class_name=cname,
            original_confidence=orig_conf,
            calibrated_confidence=orig_conf,
            # No conflict: a box without explicit flags is trusted, not weak.
            # The old defaults were inverted (trusted=False, weak=True).
            trusted=bool(target_box.get("trusted", True)),
            weak=bool(target_box.get("weak", False)),
            confidence_missing=conf_missing,
            demoted_reason=target_box.get("demoted_reason"),
            box=target_box,
        )

    def validate_boxes(
        self,
        stem_or_path: Union[str, Path],
        boxes: Sequence[Dict[str, Any]],
        *,
        in_place: bool = True,
    ) -> List[Dict[str, Any]]:
        """Validate a sequence of bounding boxes for a given stem.

        The stem is normalized and its materials extracted once for the
        whole batch instead of once per box; each box is then evaluated
        against the precomputed stem data (`_evaluate_box_prepared`).
        When no stem is given, boxes fall back to their own `"stem"` key,
        so the per-box path is used unchanged.
        """
        validated: List[Dict[str, Any]] = []
        stem_str = str(stem_or_path or "")
        if stem_str.strip():
            norm_stem = self.normalize_stem(stem_str)
            stem_mats = self.extract_stem_materials(norm_stem)
            for b in boxes or []:
                if not isinstance(b, dict):
                    continue
                target = b if in_place else dict(b)
                validated.append(
                    self._evaluate_box_prepared(norm_stem, stem_mats, target).box
                )
            return validated
        for b in boxes or []:
            if not isinstance(b, dict):
                continue
            validated.append(self.validate_box(stem_or_path, b, in_place=in_place))
        return validated

    def validate_review_payload(
        self,
        payload: Dict[str, Any],
        stem: Optional[Union[str, Path]] = None,
        *,
        in_place: bool = True,
    ) -> Dict[str, Any]:
        """Validate a review JSON payload (`trusted` and `weak` lists).

        Any box in `trusted` that has a stem material conflict is demoted (`confidence <= 0.45`,
        `trusted=False`, `weak=True`, `demoted_reason="stem_material_conflict"`) and moved to `weak`.
        Any conflicting box already in `weak` is also demoted in-place.

        S359 (v19/P3-8): dict-dışı payload artık sessizce geri
        dönmüyor — imza `Dict[str, Any]` diyor, çağıran "doğrulandı"
        sanmasın diye TypeError fırlatır.
        """
        if not isinstance(payload, dict):
            raise TypeError(
                "validate_review_payload expects a dict payload, "
                f"got {type(payload).__name__}"
            )

        target = payload if in_place else dict(payload)
        resolved_stem = str(stem or target.get("stem") or "")

        # Non-dict entries are not boxes: keep them in their list instead of
        # silently dropping them.
        trusted_in = target.get("trusted") or []
        weak_in = target.get("weak") or []

        # Hoist the stem work (normalize + material extraction) once for the
        # whole payload; per-box evaluation reuses it. With an empty stem,
        # boxes fall back to their own stem key, so the per-box path applies.
        use_prepared = bool(resolved_stem.strip())
        if use_prepared:
            norm_stem = self.normalize_stem(resolved_stem)
            stem_mats = self.extract_stem_materials(norm_stem)

        def _evaluate(b: Dict[str, Any]) -> StemValidationResult:
            if use_prepared:
                target_box = b if in_place else dict(b)
                return self._evaluate_box_prepared(norm_stem, stem_mats, target_box)
            return self.evaluate_box(resolved_stem, b, in_place=in_place)

        new_trusted: List[Dict[str, Any]] = []
        demoted_to_weak: List[Dict[str, Any]] = []

        for b in trusted_in:
            if not isinstance(b, dict):
                new_trusted.append(b)
                continue
            res = _evaluate(b)
            if res.conflict:
                demoted_to_weak.append(res.box)
            else:
                new_trusted.append(res.box)

        new_weak: List[Dict[str, Any]] = list(demoted_to_weak)
        for b in weak_in:
            if not isinstance(b, dict):
                new_weak.append(b)
                continue
            res = _evaluate(b)
            new_weak.append(res.box)

        target["trusted"] = new_trusted
        target["weak"] = new_weak
        return target

    # Aliases for convenience across callers
    validate = validate_box
    validate_annotation = validate_box
    validate_annotations = validate_boxes
    has_conflict = is_conflict
    check_conflict = is_conflict

    @classmethod
    def _unpack_conflict_args(
        cls,
        stem_or_path: Union[str, Path, Dict[str, Any]],
        predicted_class: Union[str, int, Dict[str, Any], None],
        class_id: Optional[int],
    ) -> Tuple[str, str, int]:
        if isinstance(stem_or_path, dict):
            box = stem_or_path
            stem_str = str(
                predicted_class
                if isinstance(predicted_class, (str, Path))
                else (box.get("stem") or box.get("filename") or box.get("file") or "")
            )
            cid = cls._extract_class_id(box)
            cname = cls._extract_class_name(box, cid)
            return stem_str, cname, cid

        stem_str = str(stem_or_path or "")
        if isinstance(predicted_class, dict):
            cid = cls._extract_class_id(predicted_class)
            cname = cls._extract_class_name(predicted_class, cid)
            return stem_str, cname, cid

        cid = -1
        cname = ""
        if isinstance(predicted_class, int) and not isinstance(predicted_class, bool):
            cid = int(predicted_class)
        elif isinstance(predicted_class, str):
            s = predicted_class.strip()
            try:
                cid = int(s)
            except ValueError:
                cname = s

        if class_id is not None and not isinstance(class_id, bool):
            try:
                cid = int(class_id)
            except (TypeError, ValueError):
                pass

        if not cname and cid >= 0:
            cname = cls._lookup_class_name(cid)

        return stem_str, cname, cid

    @classmethod
    def _resolve_stem_and_box(
        cls,
        stem_or_box: Union[str, Path, Dict[str, Any]],
        box_or_stem: Union[Dict[str, Any], str, Path, None],
    ) -> Tuple[str, Dict[str, Any]]:
        if isinstance(stem_or_box, dict):
            box = stem_or_box
            stem_str = (
                str(box_or_stem)
                if isinstance(box_or_stem, (str, Path)) and str(box_or_stem).strip()
                else str(
                    box.get("stem")
                    or box.get("filename")
                    or box.get("file")
                    or box.get("image_path")
                    or ""
                )
            )
            return stem_str, box

        if isinstance(box_or_stem, dict):
            stem_str = str(
                stem_or_box
                or box_or_stem.get("stem")
                or box_or_stem.get("filename")
                or box_or_stem.get("file")
                or ""
            )
            return stem_str, box_or_stem

        raise TypeError("validate_box requires a box dictionary and an optional stem string/Path.")

    @staticmethod
    def _extract_class_id(box: Mapping[str, Any]) -> int:
        raw = box.get("class_id", box.get("cls", box.get("id", -1)))
        if isinstance(raw, bool) or raw is None:
            return -1
        try:
            return int(raw)
        except (TypeError, ValueError):
            return -1

    @classmethod
    def _extract_class_name(cls, box: Mapping[str, Any], class_id: int = -1) -> str:
        for key in ("class_name", "label", "name", "key", "material"):
            val = box.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        if class_id >= 0:
            return cls._lookup_class_name(class_id)
        return ""

    @staticmethod
    def _lookup_class_name(class_id: int) -> str:
        try:
            from dataset_factory.class_taxonomy import dataset_class_names

            names = dataset_class_names()
            if int(class_id) in names:
                return str(names[int(class_id)])
        except Exception:
            pass
        # Canonical fallback (mirrors dataset_class_names(), 0..26). The old
        # table covered only 10 ids, so _lookup_class_name() returned
        # "class_<id>" for every other canonical class when the taxonomy
        # import was unavailable.
        fallback = {
            0: "herb_aria",
            1: "herb_mela",
            2: "herb_targena",
            3: "ore_orichalcum",
            4: "gem_crystal",
            5: "tree_log",
            6: "herb_edana",
            7: "herb_lumina",
            8: "ore_obsidian",
            9: "minimap_sprout",
            10: "monster_3d",
            11: "minimap_monster_dot",
            12: "crystal_blood",
            13: "skill_ui_prompt",
            14: "obstacle",
            15: "path_ground",
            16: "player_hp_bar",
            17: "player_mp_bar",
            18: "target_hp_bar",
            19: "skill_bar",
            20: "skill_slot_ready",
            21: "skill_slot_cooldown",
            22: "hp_potion",
            23: "mp_potion",
            24: "buff_scroll",
            25: "kinah",
            26: "player_character",
        }
        return fallback.get(int(class_id), f"class_{int(class_id)}")

    @staticmethod
    def _extract_confidence(box: Mapping[str, Any]) -> Optional[float]:
        """Kutunun olculmus guvenini doner; guven alani yoksa/ayristirilamazsa
        None doner (B-09: olculmemis guvene 0.40 gibi uydurma varsayilan
        yazilmaz — eksiklik confidence_missing ile isaretlenir)."""
        for key in ("confidence", "conf", "score"):
            if key in box and box[key] is not None:
                try:
                    return float(box[key])
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _is_protected_hand_label(box: Mapping[str, Any]) -> bool:
        return bool(
            box.get("hand_label") is True
            or box.get("human_approved") is True
            or box.get("approved") is True
            or box.get("confirmed") is True
        )


_DEFAULT_VALIDATOR = StemMaterialValidator()


def extract_stem_material_hints(stem_or_path: Union[str, Path, None]) -> Set[str]:
    """Module-level helper to extract canonical material hints from a stem or filename."""
    return _DEFAULT_VALIDATOR.extract_stem_materials(stem_or_path)


def is_stem_material_conflict(
    stem_or_path: Union[str, Path, Dict[str, Any]],
    predicted_class: Union[str, int, Dict[str, Any], None] = None,
    class_id: Optional[int] = None,
) -> bool:
    """Module-level helper to check if a stem's material hint conflicts with predicted class."""
    return _DEFAULT_VALIDATOR.is_conflict(stem_or_path, predicted_class, class_id=class_id)


def validate_and_demote_box(
    stem_or_box: Union[str, Path, Dict[str, Any]],
    box_or_stem: Union[Dict[str, Any], str, Path, None] = None,
    *,
    in_place: bool = True,
) -> Dict[str, Any]:
    """Module-level helper to validate and demote a single box on stem material conflict."""
    return _DEFAULT_VALIDATOR.validate_box(stem_or_box, box_or_stem, in_place=in_place)


__all__ = [
    "CLASS_ID_TO_MATERIAL_FAMILY",
    "DEMOTED_REASON_STEM_MATERIAL_CONFLICT",
    "MATERIAL_FAMILY_SYNONYMS",
    "MAX_CONFLICT_CONFIDENCE",
    "StemMaterialValidator",
    "StemValidationResult",
    "TOKEN_TO_MATERIAL_FAMILY",
    "extract_stem_material_hints",
    "is_stem_material_conflict",
    "validate_and_demote_box",
]
