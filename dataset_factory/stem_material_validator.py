# -*- coding: utf-8 -*-
"""Stem Material Validator for Dataset Factory (`dataset_factory/stem_material_validator.py`).

Compares material hints embedded in image/label filenames or stems
(e.g. `boot_*_aria_*`, `mela`, `targena`, `orichalcum`, `obsidian`, `wood`)
against the predicted class of a bounding box. When a conflict is detected,
demotes the box confidence to `<= 0.45` and sets:
    `trusted = False`
    `weak = True`
    `demoted_reason = "stem_material_conflict"`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union


MAX_CONFLICT_CONFIDENCE: float = 0.45
DEFAULT_FALLBACK_CONFIDENCE: float = 0.40
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
    original_confidence: float
    calibrated_confidence: float
    trusted: bool
    weak: bool
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
            "demoted_reason": self.demoted_reason,
            "box": dict(self.box),
        }


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
        default_fallback_confidence: float = DEFAULT_FALLBACK_CONFIDENCE,
        demoted_reason: str = DEMOTED_REASON_STEM_MATERIAL_CONFLICT,
        protect_hand_labels: bool = False,
    ) -> None:
        if max_conflict_confidence > MAX_CONFLICT_CONFIDENCE or max_conflict_confidence <= 0.0:
            raise ValueError(
                f"max_conflict_confidence must be in (0.0, {MAX_CONFLICT_CONFIDENCE}], "
                f"got {max_conflict_confidence}"
            )
        self.max_conflict_confidence: float = float(max_conflict_confidence)
        self.default_fallback_confidence: float = min(
            float(default_fallback_confidence), self.max_conflict_confidence
        )
        self.demoted_reason: str = str(demoted_reason or DEMOTED_REASON_STEM_MATERIAL_CONFLICT)
        self.protect_hand_labels: bool = bool(protect_hand_labels)

    @staticmethod
    def normalize_stem(stem_or_path: Union[str, Path, None]) -> str:
        """Extract normalized lowercase stem from a filename, path, or stem string."""
        if stem_or_path is None:
            return ""
        raw = str(stem_or_path).strip()
        if not raw:
            return ""
        name = Path(raw).name
        lower_name = name.lower()
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".json", ".txt"):
            if lower_name.endswith(ext):
                lower_name = lower_name[: -len(ext)]
                break
        return lower_name.strip()

    @classmethod
    def extract_stem_materials(cls, stem_or_path: Union[str, Path, None]) -> Set[str]:
        """Extract canonical material families hinted by the filename/stem.

        Handles patterns like `boot_001_aria_node`, `ref_mela_02`, `targena`,
        `orichalcum`, `obsidian`, `wood`, `herb_edana`, `crystal_blood`, etc.
        """
        s_lower = cls.normalize_stem(stem_or_path)
        if not s_lower:
            return set()

        found: Set[str] = set()

        # Check explicit boot_*_<material>_* pattern first
        boot_match = re.findall(
            r"boot(?:_[a-z0-9]+)*?_(aria|mela|targena|orichalcum|obsidian|wood|tree|log|crystal|gem|edana|lumina|blood)(?:_|$)",
            s_lower,
        )
        for token in boot_match:
            fam = TOKEN_TO_MATERIAL_FAMILY.get(token)
            if fam:
                found.add(fam)

        # Compound blood crystal check before generic crystal check
        has_blood_crystal = "blood_crystal" in s_lower or "crystal_blood" in s_lower
        if has_blood_crystal:
            found.add("blood")

        # Tokenize stem on non-alphanumeric separators
        tokens = [t for t in re.split(r"[^a-z0-9]+", s_lower) if t]
        for tok in tokens:
            if tok == "crystal" and has_blood_crystal:
                continue
            fam = TOKEN_TO_MATERIAL_FAMILY.get(tok)
            if fam is not None:
                found.add(fam)

        # Substring fallback for concatenated stems (e.g., `boot01aria`, `orichalcum02`)
        if not found:
            for key_token, fam in (
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
            ):
                if key_token in s_lower:
                    found.add(fam)

        return found

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

            tokens = [t for t in re.split(r"[^a-z0-9]+", c_lower) if t]
            for tok in tokens:
                if tok == "crystal" and has_blood_crystal:
                    continue
                fam = TOKEN_TO_MATERIAL_FAMILY.get(tok)
                if fam is not None:
                    found.add(fam)

            if not found:
                for key_token, fam in TOKEN_TO_MATERIAL_FAMILY.items():
                    if len(key_token) >= 4 and key_token in c_lower:
                        if key_token == "crystal" and has_blood_crystal:
                            continue
                        found.add(fam)

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

        # Raw essence class or out-of-taxonomy class ID (>= 25)
        if "essence" in c_lower or c_lower.startswith("essence_") or cid >= 25:
            # If essence has a material token that matches the stem material, check if it conflicts
            box_mats = cls.extract_box_materials(c_lower, cid)
            if mats and box_mats and (mats & box_mats):
                # Still an unmapped essence class on a 3D frame if cid >= 25 or raw essence_*
                return True
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

        if not stem_mats:
            return False

        box_mats = self.extract_box_materials(cname, cid)
        if box_mats and not (stem_mats & box_mats):
            return True

        return False

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

        cid = self._extract_class_id(target_box)
        cname = self._extract_class_name(target_box, cid)
        orig_conf = self._extract_confidence(target_box)

        norm_stem = self.normalize_stem(stem_str)
        stem_mats = self.extract_stem_materials(norm_stem)
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
                demoted_reason=target_box.get("demoted_reason"),
                box=target_box,
            )

        conflict = False
        if self.is_invalid_essence_on_material_or_3d_stem(norm_stem, cname, cid, stem_mats):
            conflict = True
        elif stem_mats and box_mats and not (stem_mats & box_mats):
            conflict = True

        if conflict:
            demoted_conf = round(min(orig_conf, self.max_conflict_confidence), 4)
            if demoted_conf <= 0.0:
                demoted_conf = round(self.default_fallback_confidence, 4)
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
            trusted=bool(target_box.get("trusted", False)),
            weak=bool(target_box.get("weak", True)),
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
        """Validate a sequence of bounding boxes for a given stem."""
        validated: List[Dict[str, Any]] = []
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
        """
        if not isinstance(payload, dict):
            return payload

        target = payload if in_place else dict(payload)
        resolved_stem = str(stem or target.get("stem") or "")

        trusted_in = [b for b in (target.get("trusted") or []) if isinstance(b, dict)]
        weak_in = [b for b in (target.get("weak") or []) if isinstance(b, dict)]

        new_trusted: List[Dict[str, Any]] = []
        demoted_to_weak: List[Dict[str, Any]] = []

        for b in trusted_in:
            res = self.evaluate_box(resolved_stem, b, in_place=in_place)
            if res.conflict:
                demoted_to_weak.append(res.box)
            else:
                new_trusted.append(res.box)

        new_weak: List[Dict[str, Any]] = list(demoted_to_weak)
        for b in weak_in:
            res = self.evaluate_box(resolved_stem, b, in_place=in_place)
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
            12: "crystal_blood",
        }
        return fallback.get(int(class_id), f"class_{int(class_id)}")

    @staticmethod
    def _extract_confidence(box: Mapping[str, Any]) -> float:
        for key in ("confidence", "conf", "score"):
            if key in box and box[key] is not None:
                try:
                    val = float(box[key])
                    return val
                except (TypeError, ValueError):
                    continue
        return DEFAULT_FALLBACK_CONFIDENCE

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
    "DEFAULT_FALLBACK_CONFIDENCE",
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
