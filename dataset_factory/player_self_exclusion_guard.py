"""
Player Self-Exclusion Guard Module.

Protects the player character and mount located at the center of the screen
(`0.42 <= cx <= 0.58` and `0.40 <= cy <= 0.62`) from being accidentally
labeled as a 3D material or 3D monster/mob during automated dataset generation,
unless verified `[F Gather]` prompt or `\\Y/` crest/arma evidence is present.

Wiring note (S-2501): this guard is currently exercised only by
tests/test_w64_player_center_exclusion.py — no production module imports it
and it is not exported from dataset_factory/__init__.py. Wiring it in needs
the package export (dataset_factory/__init__.py, owned by slice R46) plus a
production caller in the dataset pipeline (e.g. yolo_auto_labeler); both are
outside this slice, so that part is recorded blocked — cross-slice.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from dataset_factory.class_taxonomy import CANONICAL_NAMES

logger = logging.getLogger(__name__)


DEFAULT_CX_MIN: float = 0.42
DEFAULT_CX_MAX: float = 0.58
DEFAULT_CY_MIN: float = 0.40
DEFAULT_CY_MAX: float = 0.62

GATHER_PROMPT_TOKENS: Tuple[str, ...] = (
    "[f gather]",
    "f gather",
    "gather_prompt",
    "f_gather",
)

CREST_EVIDENCE_TOKENS: Tuple[str, ...] = (
    "\\y/",
    "crest",
    "arma",
    "mob_crest",
    "target_crest",
)

PROTECTED_3D_CATEGORIES: Tuple[str, ...] = (
    "3d_material",
    "3d_monster",
    "3d_mob",
    "material",
    "monster",
    "mob",
    "resource",
    "ore",
    "herb",
    "wood",
    "tree",
    "vein",
    "node",
)


# Precompiled token splitter: is_protected_3d_category runs on every
# candidate / YOLO line, and re.split recompiles the pattern on each call.
_TOKEN_SPLIT_RE: re.Pattern = re.compile(r"[^a-z0-9]+")


def _protected_3d_tokens() -> frozenset:
    """Token set of PROTECTED_3D_CATEGORIES (split on non-alphanumerics)."""
    tokens = set()
    for entry in PROTECTED_3D_CATEGORIES:
        tokens.update(tok for tok in _TOKEN_SPLIT_RE.split(entry.lower()) if tok)
    return frozenset(tokens)


_PROTECTED_3D_TOKENS: frozenset = _protected_3d_tokens()


@lru_cache(maxsize=256)
def is_protected_3d_category(category: str) -> bool:
    """True if `category` names a protected 3D class.

    F-0983: the old substring check (``tok in low_cat``) over-captured, e.g.
    ``"wooden_chest"`` matched ``"wood"`` and ``"monster_hunter"`` matched
    ``"monster"``. Matching is now token-exact: the category is split on
    separators and a token must equal a protected token (``"ore_orichalcum"``
    -> ``"ore"``; ``"monster_3d"`` -> ``"monster"``/``"3d"``).
    """
    if not category:
        return False
    tokens = [tok for tok in _TOKEN_SPLIT_RE.split(category.strip().lower()) if tok]
    return any(tok in _PROTECTED_3D_TOKENS for tok in tokens)


@dataclass(frozen=True)
class ExclusionGuardConfig:
    """Configuration for the center-screen player & mount exclusion zone."""

    cx_min: float = DEFAULT_CX_MIN
    cx_max: float = DEFAULT_CX_MAX
    cy_min: float = DEFAULT_CY_MIN
    cy_max: float = DEFAULT_CY_MAX
    min_gather_confidence: float = 0.50
    min_crest_confidence: float = 0.50
    # F-0983: the category gate must be ON by default; with False every
    # center label was excluded regardless of class, so the gate did nothing.
    strict_3d_only: bool = True

    def validate(self) -> bool:
        return (
            0.0 <= self.cx_min < self.cx_max <= 1.0
            and 0.0 <= self.cy_min < self.cy_max <= 1.0
            and 0.0 <= self.min_gather_confidence <= 1.0
            and 0.0 <= self.min_crest_confidence <= 1.0
        )


@dataclass(frozen=True)
class ExclusionDecision:
    """Result of evaluating a single candidate bounding box or annotation."""

    excluded: bool
    in_exclusion_zone: bool
    has_gather_evidence: bool
    has_crest_evidence: bool
    normalized_cx: float
    normalized_cy: float
    reason: str
    category: str = ""

    @property
    def allowed(self) -> bool:
        return not self.excluded

    def to_dict(self) -> Dict[str, Any]:
        return {
            "excluded": self.excluded,
            "allowed": self.allowed,
            "in_exclusion_zone": self.in_exclusion_zone,
            "has_gather_evidence": self.has_gather_evidence,
            "has_crest_evidence": self.has_crest_evidence,
            "normalized_cx": round(self.normalized_cx, 6),
            "normalized_cy": round(self.normalized_cy, 6),
            "reason": self.reason,
            "category": self.category,
        }


class PlayerSelfExclusionGuard:
    """
    Guards automated dataset labeling from misclassifying the player's own
    character or mount at the screen center (`0.42 <= cx <= 0.58`,
    `0.40 <= cy <= 0.62`) as a 3D material or 3D monster.

    Candidates inside the exclusion zone are rejected unless backed by true
    `[F Gather]` interaction prompt evidence or `\\Y/` monster crest evidence.
    """

    def __init__(self, config: Optional[ExclusionGuardConfig] = None) -> None:
        self.config: ExclusionGuardConfig = config or ExclusionGuardConfig()
        if not self.config.validate():
            raise ValueError(f"Invalid ExclusionGuardConfig: {self.config}")
        self._total_evaluated: int = 0
        self._total_excluded: int = 0
        self._total_overridden_by_evidence: int = 0
        self._yolo_lines_skipped_malformed: int = 0
        self._yolo_lines_skipped_unparsable: int = 0
        # S-2503: candidates skipped by filter_candidates because their
        # center could not be extracted/parsed (counted, not fatal).
        self._candidates_skipped_unparsable: int = 0

    @staticmethod
    def _coerce_confidence(value: Any) -> Optional[float]:
        """Safely coerce a confidence value to float; non-numeric -> None.

        F-0883: a non-numeric confidence in a candidate dict used to crash
        float() and take the whole filter down. Garbage is treated as
        "no confidence info" (same as None), never as evidence, never a crash.
        """
        if value is None or isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return `(cx_min, cx_max, cy_min, cy_max)`."""
        return (
            self.config.cx_min,
            self.config.cx_max,
            self.config.cy_min,
            self.config.cy_max,
        )

    @staticmethod
    def normalize_center(
        cx: float,
        cy: float,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Convert center coordinates to normalized `[0.0, 1.0]` space if needed.

        F-0980: a pixel-scale coordinate (> 1.0) whose frame dimension is
        missing used to pass through silently, un-normalized. That is now a
        ValueError instead of a quiet wrong answer.
        """
        norm_x = float(cx)
        norm_y = float(cy)

        if norm_x > 1.0:
            if frame_width is None or frame_width <= 0:
                raise ValueError(
                    f"Pixel-scale cx={cx!r} needs frame_width to normalize (got {frame_width!r})"
                )
            norm_x = norm_x / float(frame_width)
        if norm_y > 1.0:
            if frame_height is None or frame_height <= 0:
                raise ValueError(
                    f"Pixel-scale cy={cy!r} needs frame_height to normalize (got {frame_height!r})"
                )
            norm_y = norm_y / float(frame_height)

        return norm_x, norm_y

    def _is_in_zone_norm(self, norm_x: float, norm_y: float) -> bool:
        """Zone check on already-normalized coordinates (no re-normalizing)."""
        return (
            self.config.cx_min <= norm_x <= self.config.cx_max
            and self.config.cy_min <= norm_y <= self.config.cy_max
        )

    def is_in_self_exclusion_zone(
        self,
        cx: float,
        cy: float,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> bool:
        """
        Return True if `(cx, cy)` falls within the player/mount exclusion zone:
        `0.42 <= cx <= 0.58` and `0.40 <= cy <= 0.62` (by default).
        """
        norm_x, norm_y = self.normalize_center(cx, cy, frame_width, frame_height)
        return self._is_in_zone_norm(norm_x, norm_y)

    def has_gather_evidence(
        self,
        candidate: Optional[ Union[Mapping[str, Any], Any] ] = None,
        *,
        f_gather_detected: bool = False,
        gather_confidence: Optional[float] = None,
        ocr_text: Optional[str] = None,
        evidence_tags: Optional[Iterable[str]] = None,
        _candidate_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check whether verified `[F Gather]` evidence exists for the candidate.

        F-0808: a bare confidence number is NOT evidence on its own. Only a
        real detection (``f_gather_detected`` / a ``has_f_gather``-style flag),
        OCR prompt text, or an evidence tag counts; the confidence only
        qualifies evidence that already exists.
        """
        if f_gather_detected:
            conf = self._coerce_confidence(gather_confidence)
            if conf is None or conf >= self.config.min_gather_confidence:
                return True

        if ocr_text:
            lowered_ocr = ocr_text.strip().lower()
            if any(token in lowered_ocr for token in GATHER_PROMPT_TOKENS):
                return True

        if evidence_tags:
            for tag in evidence_tags:
                low_tag = str(tag).strip().lower()
                if any(token in low_tag for token in GATHER_PROMPT_TOKENS):
                    return True

        if candidate is not None or _candidate_data is not None:
            data = _candidate_data if _candidate_data is not None else self._extract_mapping(candidate)
            if bool(data.get("has_f_gather")) or bool(data.get("f_gather")) or bool(data.get("gather_verified")):
                # F-0883: coerce safely; non-numeric confidence must not crash the filter.
                conf = self._coerce_confidence(data.get("gather_confidence", data.get("f_gather_conf")))
                if conf is None or conf >= self.config.min_gather_confidence:
                    return True

            for text_key in ("ocr_text", "prompt_text", "evidence_text", "ui_prompt"):
                val = data.get(text_key)
                if isinstance(val, str) and any(tok in val.lower() for tok in GATHER_PROMPT_TOKENS):
                    return True

            tags = data.get("evidence_tags") or data.get("tags") or ()
            if isinstance(tags, (list, tuple, set)):
                for tag in tags:
                    if any(tok in str(tag).lower() for tok in GATHER_PROMPT_TOKENS):
                        return True

        return False

    def has_crest_evidence(
        self,
        candidate: Optional[ Union[Mapping[str, Any], Any] ] = None,
        *,
        crest_detected: bool = False,
        crest_confidence: Optional[float] = None,
        ocr_text: Optional[str] = None,
        evidence_tags: Optional[Iterable[str]] = None,
        _candidate_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check whether verified `\\Y/` crest/arma evidence exists for the candidate.

        Same rule as gather evidence (F-0808): a bare confidence number is
        NOT evidence on its own; only a real detection, crest text, or a
        crest tag counts, with confidence only qualifying that evidence.
        """
        if crest_detected:
            conf = self._coerce_confidence(crest_confidence)
            if conf is None or conf >= self.config.min_crest_confidence:
                return True

        if ocr_text:
            lowered_ocr = ocr_text.strip().lower()
            if any(token in lowered_ocr for token in CREST_EVIDENCE_TOKENS):
                return True

        if evidence_tags:
            for tag in evidence_tags:
                low_tag = str(tag).strip().lower()
                if any(token in low_tag for token in CREST_EVIDENCE_TOKENS):
                    return True

        if candidate is not None or _candidate_data is not None:
            data = _candidate_data if _candidate_data is not None else self._extract_mapping(candidate)
            if (
                bool(data.get("has_crest"))
                or bool(data.get("crest_verified"))
                or bool(data.get("has_arma"))
                or bool(data.get("arma_verified"))
                or bool(data.get("has_y_crest"))
            ):
                # F-0883: coerce safely; non-numeric confidence must not crash the filter.
                conf = self._coerce_confidence(data.get("crest_confidence", data.get("arma_confidence")))
                if conf is None or conf >= self.config.min_crest_confidence:
                    return True

            for text_key in ("ocr_text", "crest_symbol", "evidence_text", "arma_text"):
                val = data.get(text_key)
                if isinstance(val, str) and any(tok in val.lower() for tok in CREST_EVIDENCE_TOKENS):
                    return True

            tags = data.get("evidence_tags") or data.get("tags") or ()
            if isinstance(tags, (list, tuple, set)):
                for tag in tags:
                    if any(tok in str(tag).lower() for tok in CREST_EVIDENCE_TOKENS):
                        return True

        return False

    def evaluate(
        self,
        cx: float,
        cy: float,
        *,
        category: str = "3d_material",
        f_gather_detected: bool = False,
        crest_detected: bool = False,
        gather_confidence: Optional[float] = None,
        crest_confidence: Optional[float] = None,
        ocr_text: Optional[str] = None,
        evidence_tags: Optional[Iterable[str]] = None,
        candidate: Optional[Any] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        _candidate_data: Optional[Dict[str, Any]] = None,
    ) -> ExclusionDecision:
        """
        Evaluate whether a candidate at `(cx, cy)` should be excluded to protect
        the player character and mount from false 3D material/monster labeling.

        ``_candidate_data`` is an internal fast path: a mapping already
        extracted from ``candidate`` (by ``evaluate_candidate``). Passing it
        avoids re-extracting the same candidate dict once per evidence check.
        """
        self._total_evaluated += 1
        norm_x, norm_y = self.normalize_center(cx, cy, frame_width, frame_height)
        in_zone = self._is_in_zone_norm(norm_x, norm_y)

        if _candidate_data is None and candidate is not None:
            _candidate_data = self._extract_mapping(candidate)

        gather_ok = self.has_gather_evidence(
            candidate,
            f_gather_detected=f_gather_detected,
            gather_confidence=gather_confidence,
            ocr_text=ocr_text,
            evidence_tags=evidence_tags,
            _candidate_data=_candidate_data,
        )
        crest_ok = self.has_crest_evidence(
            candidate,
            crest_detected=crest_detected,
            crest_confidence=crest_confidence,
            ocr_text=ocr_text,
            evidence_tags=evidence_tags,
            _candidate_data=_candidate_data,
        )

        if not in_zone:
            return ExclusionDecision(
                excluded=False,
                in_exclusion_zone=False,
                has_gather_evidence=gather_ok,
                has_crest_evidence=crest_ok,
                normalized_cx=norm_x,
                normalized_cy=norm_y,
                reason="outside_self_exclusion_zone",
                category=category,
            )

        if self.config.strict_3d_only and category:
            # F-0983: token-exact match (see is_protected_3d_category); the
            # old substring check over-captured ("wooden_chest" ~ "wood").
            if not is_protected_3d_category(category):
                return ExclusionDecision(
                    excluded=False,
                    in_exclusion_zone=True,
                    has_gather_evidence=gather_ok,
                    has_crest_evidence=crest_ok,
                    normalized_cx=norm_x,
                    normalized_cy=norm_y,
                    reason="non_3d_category_exempt",
                    category=category,
                )

        if gather_ok or crest_ok:
            self._total_overridden_by_evidence += 1
            reason = (
                "allowed_by_f_gather_and_crest_evidence"
                if (gather_ok and crest_ok)
                else ("allowed_by_f_gather_evidence" if gather_ok else "allowed_by_crest_evidence")
            )
            return ExclusionDecision(
                excluded=False,
                in_exclusion_zone=True,
                has_gather_evidence=gather_ok,
                has_crest_evidence=crest_ok,
                normalized_cx=norm_x,
                normalized_cy=norm_y,
                reason=reason,
                category=category,
            )

        self._total_excluded += 1
        return ExclusionDecision(
            excluded=True,
            in_exclusion_zone=True,
            has_gather_evidence=False,
            has_crest_evidence=False,
            normalized_cx=norm_x,
            normalized_cy=norm_y,
            reason="excluded_player_mount_center_zone_without_evidence",
            category=category,
        )

    def should_exclude(
        self,
        cx: float,
        cy: float,
        *,
        category: str = "3d_material",
        f_gather_detected: bool = False,
        crest_detected: bool = False,
        gather_confidence: Optional[float] = None,
        crest_confidence: Optional[float] = None,
        ocr_text: Optional[str] = None,
        evidence_tags: Optional[Iterable[str]] = None,
        candidate: Optional[Any] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> bool:
        """Return True if the candidate at `(cx, cy)` must be excluded.

        F-0981: `category` was missing here, so the `strict_3d_only`
        exemption was unreachable through this API.
        """
        decision = self.evaluate(
            cx,
            cy,
            category=category,
            f_gather_detected=f_gather_detected,
            crest_detected=crest_detected,
            gather_confidence=gather_confidence,
            crest_confidence=crest_confidence,
            ocr_text=ocr_text,
            evidence_tags=evidence_tags,
            candidate=candidate,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        return decision.excluded

    def evaluate_candidate(
        self,
        candidate: Union[Mapping[str, Any], Sequence[float], Any],
        *,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> ExclusionDecision:
        """Extract `(cx, cy)` and metadata from a candidate object/dict/tuple and evaluate it.

        F-0982: frame-level ``f_gather_detected``/``crest_detected`` flags were
        applied to EVERY candidate, so one prompt freed all center boxes
        (including the player's own). Evidence now comes only from the
        candidate itself (its flags, confidence, OCR text, or tags), which
        ``evaluate`` already reads from ``candidate``.
        """
        # Extract the candidate mapping once and reuse it for the center,
        # the category and the evidence checks (evaluate -> has_*_evidence).
        data = self._extract_mapping(candidate)
        cx, cy = self._extract_center(candidate, frame_width=frame_width, frame_height=frame_height, _data=data)
        category = str(data.get("category") or data.get("label") or data.get("class_name") or "3d_material")
        return self.evaluate(
            cx,
            cy,
            category=category,
            candidate=candidate,
            frame_width=frame_width,
            frame_height=frame_height,
            _candidate_data=data,
        )

    def filter_candidates(
        self,
        candidates: Iterable[Any],
        *,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> List[Any]:
        """Filter an iterable of candidate detections, keeping only non-excluded ones.

        F-0982: no frame-level evidence flags here anymore; each candidate is
        judged on its own evidence (see evaluate_candidate).

        S-2503: one unparsable candidate must not kill the whole batch.
        Candidates whose center cannot be extracted (non-numeric / missing
        coordinates, pixel-scale without frame dims) are skipped, counted
        (see get_diagnostics "candidates_skipped_unparsable") and logged;
        every other candidate is still evaluated normally.
        """
        kept: List[Any] = []
        for item in candidates:
            try:
                decision = self.evaluate_candidate(
                    item,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            except (ValueError, TypeError) as exc:
                self._candidates_skipped_unparsable += 1
                logger.warning(
                    "filter_candidates: skipping unparsable candidate %r: %s",
                    item,
                    exc,
                )
                continue
            if decision.allowed:
                kept.append(item)
        return kept

    def filter_yolo_lines(
        self,
        yolo_lines: Iterable[str],
        *,
        f_gather_detected: bool = False,
        crest_detected: bool = False,
    ) -> List[str]:
        """
        Filter raw YOLO annotation lines (`class_id cx cy w h`) in normalized coordinates.

        F-0884: the class id is now read and mapped to its canonical category
        name, so the `strict_3d_only` gate can exempt non-3D classes (e.g.
        ``player_character`` / ``skill_ui_prompt``) instead of deleting every
        center label regardless of class. Frame-level evidence flags are kept
        here on purpose: a YOLO line carries no per-line evidence, so the
        frame's [F Gather] / crest prompt is the only signal available.
        F-1064: malformed/unparsable lines are counted and logged instead of
        being dropped silently (see diagnostics counters).
        """
        valid_lines: List[str] = []
        for raw_line in yolo_lines:
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 5:
                self._yolo_lines_skipped_malformed += 1
                logger.debug("filter_yolo_lines: skipping malformed line (<%d parts): %r", len(parts), stripped)
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                self._yolo_lines_skipped_unparsable += 1
                logger.debug("filter_yolo_lines: skipping line with non-numeric class_id: %r", stripped)
                continue
            try:
                cx = float(parts[1])
                cy = float(parts[2])
            except ValueError:
                self._yolo_lines_skipped_unparsable += 1
                logger.debug("filter_yolo_lines: skipping line with non-numeric center: %r", stripped)
                continue
            # S-5201: a non-finite center (nan/inf) used to slip through —
            # nan compares False against every bound in normalize_center/
            # evaluate, so the line passed as 'valid'. Skip just this line
            # (same counting/logging treatment as the invalid w/h below).
            if not (math.isfinite(cx) and math.isfinite(cy)):
                self._yolo_lines_skipped_unparsable += 1
                logger.debug("filter_yolo_lines: skipping line with non-finite center: %r", stripped)
                continue
            # S-2504: parts[3]/parts[4] (w/h) were never read, so a line with
            # non-numeric width/height passed as 'valid'. Parse them now; a
            # non-numeric, non-finite, or non-positive w/h is unparsable.
            try:
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                self._yolo_lines_skipped_unparsable += 1
                logger.debug("filter_yolo_lines: skipping line with non-numeric w/h: %r", stripped)
                continue
            if not (math.isfinite(w) and math.isfinite(h) and w > 0.0 and h > 0.0):
                self._yolo_lines_skipped_unparsable += 1
                logger.debug("filter_yolo_lines: skipping line with invalid w/h: %r", stripped)
                continue

            category = CANONICAL_NAMES.get(class_id, f"class_{class_id}")
            # S-5201: evaluate() is called without frame_width/frame_height,
            # so a finite pixel-scale center (e.g. cx=960) raises ValueError
            # from normalize_center (F-0980) — that used to escape and kill
            # the whole batch. Skip just this line instead.
            try:
                decision = self.evaluate(
                    cx,
                    cy,
                    category=category,
                    f_gather_detected=f_gather_detected,
                    crest_detected=crest_detected,
                )
            except ValueError:
                self._yolo_lines_skipped_unparsable += 1
                logger.debug("filter_yolo_lines: skipping line with pixel-scale center (no frame dims): %r", stripped)
                continue
            if decision.allowed:
                valid_lines.append(stripped)
        return valid_lines

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return operational metrics and zone configuration."""
        return {
            "module": "player_self_exclusion_guard",
            "cx_range": [self.config.cx_min, self.config.cx_max],
            "cy_range": [self.config.cy_min, self.config.cy_max],
            "total_evaluated": self._total_evaluated,
            "total_excluded": self._total_excluded,
            "total_overridden_by_evidence": self._total_overridden_by_evidence,
            "yolo_lines_skipped_malformed": self._yolo_lines_skipped_malformed,
            "yolo_lines_skipped_unparsable": self._yolo_lines_skipped_unparsable,
            "candidates_skipped_unparsable": self._candidates_skipped_unparsable,
            "ready": self.config.validate(),
        }

    @staticmethod
    def _extract_mapping(candidate: Any) -> Dict[str, Any]:
        if isinstance(candidate, Mapping):
            return dict(candidate)
        if hasattr(candidate, "__dict__"):
            return dict(vars(candidate))
        return {}

    @staticmethod
    def _coerce_center_float(value: Any, field: str) -> float:
        """Coerce one center/bbox coordinate to float (S-2503).

        A non-numeric value (or None) used to raise a bare TypeError from
        float() deep inside _extract_center; it now raises ValueError with
        the field name, so filter_candidates can catch one exception type
        and skip just that candidate instead of killing the whole batch.
        """
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric center coordinate {field}={value!r}") from exc

    def _extract_center(
        self,
        candidate: Any,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        _data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        if isinstance(candidate, (tuple, list)):
            if len(candidate) == 2:
                return self.normalize_center(
                    self._coerce_center_float(candidate[0], "cx"),
                    self._coerce_center_float(candidate[1], "cy"),
                    frame_width,
                    frame_height,
                )
            if len(candidate) == 4:
                return self._center_from_four_values(
                    self._coerce_center_float(candidate[0], "x1"),
                    self._coerce_center_float(candidate[1], "y1"),
                    self._coerce_center_float(candidate[2], "x2"),
                    self._coerce_center_float(candidate[3], "y2"),
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            if len(candidate) >= 5:
                return self.normalize_center(
                    self._coerce_center_float(candidate[1], "cx"),
                    self._coerce_center_float(candidate[2], "cy"),
                    frame_width,
                    frame_height,
                )

        data = _data if _data is not None else self._extract_mapping(candidate)
        if "cx" in data and "cy" in data:
            return self.normalize_center(
                self._coerce_center_float(data["cx"], "cx"),
                self._coerce_center_float(data["cy"], "cy"),
                frame_width,
                frame_height,
            )
        if "x_center" in data and "y_center" in data:
            return self.normalize_center(
                self._coerce_center_float(data["x_center"], "x_center"),
                self._coerce_center_float(data["y_center"], "y_center"),
                frame_width,
                frame_height,
            )
        if "center" in data and isinstance(data["center"], (tuple, list)) and len(data["center"]) >= 2:
            return self.normalize_center(
                self._coerce_center_float(data["center"][0], "center[0]"),
                self._coerce_center_float(data["center"][1], "center[1]"),
                frame_width,
                frame_height,
            )
        if all(k in data for k in ("x1", "y1", "x2", "y2")):
            cx = (
                self._coerce_center_float(data["x1"], "x1")
                + self._coerce_center_float(data["x2"], "x2")
            ) * 0.5
            cy = (
                self._coerce_center_float(data["y1"], "y1")
                + self._coerce_center_float(data["y2"], "y2")
            ) * 0.5
            return self.normalize_center(cx, cy, frame_width, frame_height)
        if "bbox" in data and isinstance(data["bbox"], (tuple, list)) and len(data["bbox"]) == 4:
            b = data["bbox"]
            return self._center_from_four_values(
                self._coerce_center_float(b[0], "bbox[0]"),
                self._coerce_center_float(b[1], "bbox[1]"),
                self._coerce_center_float(b[2], "bbox[2]"),
                self._coerce_center_float(b[3], "bbox[3]"),
                frame_width=frame_width,
                frame_height=frame_height,
                bbox_format=str(data.get("bbox_format", "xywh")),
            )

        raise ValueError(f"Unable to extract (cx, cy) center coordinates from candidate: {candidate!r}")

    def _center_from_four_values(
        self,
        v0: float,
        v1: float,
        v2: float,
        v3: float,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        bbox_format: str = "xywh",
    ) -> Tuple[float, float]:
        fmt = bbox_format.strip().lower()
        if fmt == "xyxy":
            cx = (v0 + v2) * 0.5
            cy = (v1 + v3) * 0.5
            return self.normalize_center(cx, cy, frame_width, frame_height)
        return self.normalize_center(v0, v1, frame_width, frame_height)


__all__ = [
    "DEFAULT_CX_MIN",
    "DEFAULT_CX_MAX",
    "DEFAULT_CY_MIN",
    "DEFAULT_CY_MAX",
    "ExclusionGuardConfig",
    "ExclusionDecision",
    "PlayerSelfExclusionGuard",
    "is_protected_3d_category",
]
