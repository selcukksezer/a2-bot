"""
Player Self-Exclusion Guard Module.

Protects the player character and mount located at the center of the screen
(`0.42 <= cx <= 0.58` and `0.40 <= cy <= 0.62`) from being accidentally
labeled as a 3D material or 3D monster/mob during automated dataset generation,
unless verified `[F Gather]` prompt or `\\Y/` crest/arma evidence is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


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


@dataclass(frozen=True)
class ExclusionGuardConfig:
    """Configuration for the center-screen player & mount exclusion zone."""

    cx_min: float = DEFAULT_CX_MIN
    cx_max: float = DEFAULT_CX_MAX
    cy_min: float = DEFAULT_CY_MIN
    cy_max: float = DEFAULT_CY_MAX
    min_gather_confidence: float = 0.50
    min_crest_confidence: float = 0.50
    strict_3d_only: bool = False

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
        """Convert center coordinates to normalized `[0.0, 1.0]` space if needed."""
        norm_x = float(cx)
        norm_y = float(cy)

        if frame_width is not None and frame_width > 0 and (norm_x > 1.0 or frame_height is not None):
            if norm_x > 1.0:
                norm_x = norm_x / float(frame_width)
        if frame_height is not None and frame_height > 0 and (norm_y > 1.0 or frame_width is not None):
            if norm_y > 1.0:
                norm_y = norm_y / float(frame_height)

        return norm_x, norm_y

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
        return (
            self.config.cx_min <= norm_x <= self.config.cx_max
            and self.config.cy_min <= norm_y <= self.config.cy_max
        )

    def has_gather_evidence(
        self,
        candidate: Optional[ Union[Mapping[str, Any], Any] ] = None,
        *,
        f_gather_detected: bool = False,
        gather_confidence: Optional[float] = None,
        ocr_text: Optional[str] = None,
        evidence_tags: Optional[Iterable[str]] = None,
    ) -> bool:
        """Check whether verified `[F Gather]` evidence exists for the candidate."""
        if f_gather_detected:
            if gather_confidence is None or gather_confidence >= self.config.min_gather_confidence:
                return True

        if gather_confidence is not None and gather_confidence >= self.config.min_gather_confidence:
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

        if candidate is not None:
            data = self._extract_mapping(candidate)
            if bool(data.get("has_f_gather")) or bool(data.get("f_gather")) or bool(data.get("gather_verified")):
                conf = data.get("gather_confidence", data.get("f_gather_conf"))
                if conf is None or float(conf) >= self.config.min_gather_confidence:
                    return True

            cand_conf = data.get("gather_confidence", data.get("f_gather_conf"))
            if cand_conf is not None and float(cand_conf) >= self.config.min_gather_confidence:
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
    ) -> bool:
        """Check whether verified `\\Y/` crest/arma evidence exists for the candidate."""
        if crest_detected:
            if crest_confidence is None or crest_confidence >= self.config.min_crest_confidence:
                return True

        if crest_confidence is not None and crest_confidence >= self.config.min_crest_confidence:
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

        if candidate is not None:
            data = self._extract_mapping(candidate)
            if (
                bool(data.get("has_crest"))
                or bool(data.get("crest_verified"))
                or bool(data.get("has_arma"))
                or bool(data.get("arma_verified"))
                or bool(data.get("has_y_crest"))
            ):
                conf = data.get("crest_confidence", data.get("arma_confidence"))
                if conf is None or float(conf) >= self.config.min_crest_confidence:
                    return True

            cand_conf = data.get("crest_confidence", data.get("arma_confidence"))
            if cand_conf is not None and float(cand_conf) >= self.config.min_crest_confidence:
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
    ) -> ExclusionDecision:
        """
        Evaluate whether a candidate at `(cx, cy)` should be excluded to protect
        the player character and mount from false 3D material/monster labeling.
        """
        self._total_evaluated += 1
        norm_x, norm_y = self.normalize_center(cx, cy, frame_width, frame_height)
        in_zone = self.is_in_self_exclusion_zone(norm_x, norm_y)

        gather_ok = self.has_gather_evidence(
            candidate,
            f_gather_detected=f_gather_detected,
            gather_confidence=gather_confidence,
            ocr_text=ocr_text,
            evidence_tags=evidence_tags,
        )
        crest_ok = self.has_crest_evidence(
            candidate,
            crest_detected=crest_detected,
            crest_confidence=crest_confidence,
            ocr_text=ocr_text,
            evidence_tags=evidence_tags,
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
            low_cat = category.strip().lower()
            if not any(tok in low_cat for tok in PROTECTED_3D_CATEGORIES):
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
        """Return True if the candidate at `(cx, cy)` must be excluded."""
        decision = self.evaluate(
            cx,
            cy,
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
        f_gather_detected: bool = False,
        crest_detected: bool = False,
    ) -> ExclusionDecision:
        """Extract `(cx, cy)` and metadata from a candidate object/dict/tuple and evaluate it."""
        cx, cy = self._extract_center(candidate, frame_width=frame_width, frame_height=frame_height)
        data = self._extract_mapping(candidate)
        category = str(data.get("category") or data.get("label") or data.get("class_name") or "3d_material")
        return self.evaluate(
            cx,
            cy,
            category=category,
            f_gather_detected=f_gather_detected,
            crest_detected=crest_detected,
            candidate=candidate,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    def filter_candidates(
        self,
        candidates: Iterable[Any],
        *,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        f_gather_detected: bool = False,
        crest_detected: bool = False,
    ) -> List[Any]:
        """Filter an iterable of candidate detections, keeping only non-excluded ones."""
        kept: List[Any] = []
        for item in candidates:
            decision = self.evaluate_candidate(
                item,
                frame_width=frame_width,
                frame_height=frame_height,
                f_gather_detected=f_gather_detected,
                crest_detected=crest_detected,
            )
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
        """
        valid_lines: List[str] = []
        for raw_line in yolo_lines:
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 5:
                continue
            try:
                cx = float(parts[1])
                cy = float(parts[2])
            except ValueError:
                continue

            decision = self.evaluate(
                cx,
                cy,
                f_gather_detected=f_gather_detected,
                crest_detected=crest_detected,
            )
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
            "ready": self.config.validate(),
        }

    @staticmethod
    def _extract_mapping(candidate: Any) -> Dict[str, Any]:
        if isinstance(candidate, Mapping):
            return dict(candidate)
        if hasattr(candidate, "__dict__"):
            return dict(vars(candidate))
        return {}

    def _extract_center(
        self,
        candidate: Any,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> Tuple[float, float]:
        if isinstance(candidate, (tuple, list)):
            if len(candidate) == 2:
                return self.normalize_center(float(candidate[0]), float(candidate[1]), frame_width, frame_height)
            if len(candidate) == 4:
                return self._center_from_four_values(
                    float(candidate[0]),
                    float(candidate[1]),
                    float(candidate[2]),
                    float(candidate[3]),
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            if len(candidate) >= 5:
                return self.normalize_center(float(candidate[1]), float(candidate[2]), frame_width, frame_height)

        data = self._extract_mapping(candidate)
        if "cx" in data and "cy" in data:
            return self.normalize_center(float(data["cx"]), float(data["cy"]), frame_width, frame_height)
        if "x_center" in data and "y_center" in data:
            return self.normalize_center(
                float(data["x_center"]),
                float(data["y_center"]),
                frame_width,
                frame_height,
            )
        if "center" in data and isinstance(data["center"], (tuple, list)) and len(data["center"]) >= 2:
            return self.normalize_center(
                float(data["center"][0]),
                float(data["center"][1]),
                frame_width,
                frame_height,
            )
        if all(k in data for k in ("x1", "y1", "x2", "y2")):
            cx = (float(data["x1"]) + float(data["x2"])) * 0.5
            cy = (float(data["y1"]) + float(data["y2"])) * 0.5
            return self.normalize_center(cx, cy, frame_width, frame_height)
        if "bbox" in data and isinstance(data["bbox"], (tuple, list)) and len(data["bbox"]) == 4:
            b = data["bbox"]
            return self._center_from_four_values(
                float(b[0]),
                float(b[1]),
                float(b[2]),
                float(b[3]),
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
]
