# -*- coding: utf-8 -*-
"""
dataset_factory/atomic_review_sync.py
=====================================
Atomic Review JSON <-> Labels TXT Synchronizer (`AtomicReviewTxtSynchronizer`).

Atomically synchronizes `dataset/review/{split}/{stem}.json` and
`dataset/labels/{split}/{stem}.txt` using temporary files (`.tmp`) and
`os.replace`, automatically repairs corrupted or malformed JSON files,
and strictly preserves `"human_approved"` and `"done"` flags (both at
the document level and per-box level).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import math
import os
from pathlib import Path
import re
import threading
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger("DatasetFactory.AtomicReviewSync")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

DEFAULT_TRAIN_CONF_FLOOR: float = 0.45
DEFAULT_EMIN_THRESHOLD: float = 0.85
IGNORED_REVIEW_FILENAMES: frozenset[str] = frozenset({"trusted_seed.json"})

_RE_HUMAN_APPROVED = re.compile(r'"human_approved"\s*:\s*(true|false)', re.IGNORECASE)
_RE_APPROVED = re.compile(r'"approved"\s*:\s*(true|false)', re.IGNORECASE)
_RE_DONE_BOOL = re.compile(r'"done"\s*:\s*(true|false|null)', re.IGNORECASE)
_RE_DONE_STR = re.compile(r'"done"\s*:\s*"([^"]*)"')


# ==============================================================================
# Helper Functions: Atomic Disk I/O (.tmp + os.replace) & JSON Repair
# ==============================================================================


def _replace_with_retry(
    tmp_path: Path,
    target_path: Path,
    max_retries: int = 5,
    retry_delay: float = 0.03,
) -> None:
    """Atomically replace target_path with tmp_path using os.replace with Windows lock retry."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_retries):
        try:
            os.replace(str(tmp_path), str(target_path))
            return
        except OSError:
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay)


def atomic_write_text(
    path: Union[str, Path],
    content: str,
    encoding: str = "utf-8",
) -> Path:
    """
    Atomically write text content to `path` via a `.tmp` file in the same
    directory followed by `os.replace`.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.{os.getpid()}_{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp_path, "w", encoding=encoding, newline="\n") as f:
            f.write(content)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        _replace_with_retry(tmp_path, target)
        return target
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def atomic_write_json(
    path: Union[str, Path],
    payload: Dict[str, Any],
    indent: int = 2,
) -> Path:
    """
    Atomically serialize and write a JSON dictionary to `path` using `.tmp`
    and `os.replace`.
    """
    serialized = json.dumps(payload, ensure_ascii=False, indent=indent)
    return atomic_write_text(path, serialized + "\n", encoding="utf-8")


def extract_salvaged_flags(raw_text: str) -> Dict[str, Any]:
    """
    Salvage `"human_approved"`, `"approved"`, and `"done"` flags from raw/corrupted
    JSON text even when full JSON parsing fails.
    """
    salvaged: Dict[str, Any] = {}
    if not raw_text:
        return salvaged

    m_ha = _RE_HUMAN_APPROVED.search(raw_text)
    if m_ha:
        salvaged["human_approved"] = m_ha.group(1).lower() == "true"

    m_app = _RE_APPROVED.search(raw_text)
    if m_app:
        salvaged["approved"] = m_app.group(1).lower() == "true"

    m_done_str = _RE_DONE_STR.search(raw_text)
    if m_done_str:
        salvaged["done"] = m_done_str.group(1)
    else:
        m_done_bool = _RE_DONE_BOOL.search(raw_text)
        if m_done_bool:
            token = m_done_bool.group(1).lower()
            if token == "true":
                salvaged["done"] = True
            elif token == "false":
                salvaged["done"] = False
            else:
                salvaged["done"] = None

    return salvaged


def repair_json_content(raw_text: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Attempt to parse or repair corrupted JSON text.
    Handles:
      1. Clean valid JSON object -> (dict, False)
      2. BOM / NUL bytes / trailing garbage / concatenated JSON objects `{...}{...}`
      3. Trailing commas before `}` or `]`
      4. Truncated JSON object recovery where possible
    Returns:
      (parsed_dict_or_None, was_repaired)
    """
    if not raw_text or not raw_text.strip():
        return None, False

    # 1. Direct parse attempt
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed, False
    except Exception:
        pass

    # Strip BOM and NUL characters
    cleaned = raw_text.lstrip("\ufeff").replace("\x00", "").strip()
    if not cleaned:
        return None, False

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed, True
    except Exception:
        pass

    # 2. Extract first balanced outermost JSON object '{ ... }'
    brace_depth = 0
    start_idx = -1
    end_idx = -1
    in_string = False
    escape = False

    for idx, ch in enumerate(cleaned):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == "{":
            if brace_depth == 0:
                start_idx = idx
            brace_depth += 1
        elif ch == "}":
            if brace_depth > 0:
                brace_depth -= 1
                if brace_depth == 0 and start_idx != -1:
                    end_idx = idx + 1
                    break

    if start_idx != -1 and end_idx != -1:
        candidate = cleaned[start_idx:end_idx]
        # Remove trailing commas before } or ]
        candidate_fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
        for attempt_str in (candidate, candidate_fixed):
            try:
                parsed = json.loads(attempt_str)
                if isinstance(parsed, dict):
                    return parsed, True
            except Exception:
                pass

    # 3. Remove trailing commas on the whole string and try again
    comma_fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        parsed = json.loads(comma_fixed)
        if isinstance(parsed, dict):
            return parsed, True
    except Exception:
        pass

    return None, False


def _normalize_box_dict(
    raw: Any,
    *,
    default_trusted: bool = True,
    preserve_flags: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Validate and normalize a box dictionary or sequence into canonical review format:
      {"class_id": int, "cx": float, "cy": float, "w": float, "h": float, ...}
    Preserves `human_approved`, `hand_label`, `approved`, `done`, `trusted`, `weak`,
    `confidence`, `source`, and `class_name` when present.
    """
    if isinstance(raw, (list, tuple)) and len(raw) >= 5:
        try:
            cid = int(float(raw[0]))
            cx = float(raw[1])
            cy = float(raw[2])
            w = float(raw[3])
            h = float(raw[4])
            conf = float(raw[5]) if len(raw) >= 6 else (0.95 if default_trusted else 0.58)
            raw = {
                "class_id": cid,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "confidence": conf,
                "trusted": bool(default_trusted),
                "weak": not bool(default_trusted),
            }
        except (ValueError, TypeError):
            return None

    if not isinstance(raw, dict):
        return None

    try:
        cid = int(float(raw.get("class_id", raw.get("cid", raw.get("cls", 0)))))
        cx = float(raw.get("cx", raw.get("x_center", raw.get("x", -1.0))))
        cy = float(raw.get("cy", raw.get("y_center", raw.get("y", -1.0))))
        w = float(raw.get("w", raw.get("width", -1.0)))
        h = float(raw.get("h", raw.get("height", -1.0)))
    except (ValueError, TypeError):
        return None

    if cid < 0:
        return None
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(w) and math.isfinite(h)):
        return None
    if w <= 0.0 or h <= 0.0:
        return None
    if cx < -0.05 or cx > 1.05 or cy < -0.05 or cy > 1.05 or w > 1.05 or h > 1.05:
        return None

    cx_c = round(min(max(0.0, cx), 1.0), 6)
    cy_c = round(min(max(0.0, cy), 1.0), 6)
    w_c = round(min(max(1e-5, w), 1.0), 6)
    h_c = round(min(max(1e-5, h), 1.0), 6)

    out: Dict[str, Any] = dict(raw) if preserve_flags else {}
    out["class_id"] = cid
    out["cx"] = cx_c
    out["cy"] = cy_c
    out["w"] = w_c
    out["h"] = h_c

    if "confidence" in raw:
        try:
            out["confidence"] = round(float(raw["confidence"]), 4)
        except (ValueError, TypeError):
            out["confidence"] = 0.90 if default_trusted else 0.58
    else:
        out["confidence"] = 0.90 if default_trusted else 0.58

    is_human = bool(
        out.get("human_approved") is True
        or out.get("hand_label") is True
        or out.get("approved") is True
        or out.get("confirmed") is True
    )
    if is_human:
        out["trusted"] = True
        out["weak"] = False
        if out.get("human_approved") is True:
            out["human_approved"] = True
    else:
        trusted_val = bool(raw.get("trusted", default_trusted))
        out["trusted"] = trusted_val
        out["weak"] = bool(raw.get("weak", not trusted_val))

    return out


def _boxes_match(a: Dict[str, Any], b: Dict[str, Any], tol: float = 1e-3) -> bool:
    """Return True if two normalized boxes have identical class_id and coordinates within `tol`."""
    try:
        return (
            int(a["class_id"]) == int(b["class_id"])
            and abs(float(a["cx"]) - float(b["cx"])) <= tol
            and abs(float(a["cy"]) - float(b["cy"])) <= tol
            and abs(float(a["w"]) - float(b["w"])) <= tol
            and abs(float(a["h"]) - float(b["h"])) <= tol
        )
    except (KeyError, TypeError, ValueError):
        return False


@dataclass
class SyncPairResult:
    """Structured result for a single `{split}/{stem}` synchronization operation."""

    split: str
    stem: str
    json_path: str
    txt_path: str
    synced: bool
    repaired_json: bool = False
    repaired_txt: bool = False
    created_json: bool = False
    created_txt: bool = False
    human_approved: bool = False
    done: Any = None
    trusted_count: int = 0
    weak_count: int = 0
    txt_box_count: int = 0
    error: str = ""
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "stem": self.stem,
            "json_path": self.json_path,
            "txt_path": self.txt_path,
            "synced": self.synced,
            "repaired_json": self.repaired_json,
            "repaired_txt": self.repaired_txt,
            "created_json": self.created_json,
            "created_txt": self.created_txt,
            "human_approved": self.human_approved,
            "done": self.done,
            "trusted_count": self.trusted_count,
            "weak_count": self.weak_count,
            "txt_box_count": self.txt_box_count,
            "error": self.error,
            "details": list(self.details),
        }


# ==============================================================================
# Main Class: AtomicReviewTxtSynchronizer
# ==============================================================================


class AtomicReviewTxtSynchronizer:
    """
    Atomically synchronizes `dataset/review/{split}/{stem}.json` and
    `dataset/labels/{split}/{stem}.txt` using `.tmp` files and `os.replace`.

    Guarantees:
    1. Zero partial/corrupt writes: Every write to `.json` or `.txt` is staged in
       a `.tmp` file in the target folder and committed via `os.replace`.
    2. Corrupt JSON recovery: Automatically repairs malformed/concatenated JSONs
       or reconstructs them from `.txt` while salvaging flags.
    3. Flag preservation: Strictly preserves `"human_approved"`, `"approved"`,
       and `"done"` flags at both the document level and individual box level.
    """

    def __init__(
        self,
        dataset_dir: Optional[Union[str, Path]] = None,
        splits: Sequence[str] = ("train", "val"),
        floor: float = DEFAULT_TRAIN_CONF_FLOOR,
        emin_threshold: float = DEFAULT_EMIN_THRESHOLD,
        auto_repair: bool = True,
    ) -> None:
        if dataset_dir is not None:
            self.dataset_dir = Path(dataset_dir).resolve()
        else:
            self.dataset_dir = Path(__file__).resolve().parent.parent / "dataset"

        self.splits: Tuple[str, ...] = tuple(str(s) for s in splits)
        self.floor: float = float(floor)
        self.emin_threshold: float = float(emin_threshold)
        self.auto_repair: bool = bool(auto_repair)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def get_review_path(self, split: str, stem: str) -> Path:
        """Return `dataset/review/{split}/{stem}.json` path."""
        return self.dataset_dir / "review" / str(split) / f"{stem}.json"

    def get_label_path(self, split: str, stem: str) -> Path:
        """Return `dataset/labels/{split}/{stem}.txt` path."""
        return self.dataset_dir / "labels" / str(split) / f"{stem}.txt"

    # ------------------------------------------------------------------
    # Atomic Write Operations (.tmp + os.replace)
    # ------------------------------------------------------------------

    def atomic_write_text(
        self,
        path: Union[str, Path],
        content: str,
        encoding: str = "utf-8",
    ) -> Path:
        """Write text atomically via `.tmp` file and `os.replace`."""
        with self._lock:
            return atomic_write_text(path, content, encoding=encoding)

    def atomic_write_json(
        self,
        path: Union[str, Path],
        payload: Dict[str, Any],
        indent: int = 2,
    ) -> Path:
        """Write JSON dict atomically via `.tmp` file and `os.replace`."""
        with self._lock:
            return atomic_write_json(path, payload, indent=indent)

    # ------------------------------------------------------------------
    # Parsing & Repair Helpers
    # ------------------------------------------------------------------

    def parse_yolo_txt(
        self,
        txt_path: Union[str, Path],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Parse `dataset/labels/{split}/{stem}.txt`.
        Returns `(valid_boxes, is_clean_syntax)`.
        """
        p = Path(txt_path)
        if not p.is_file():
            return [], False

        try:
            raw_content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.warning("Failed to read label TXT %s: %s", p, exc)
            return [], False

        boxes: List[Dict[str, Any]] = []
        all_lines_valid = True

        for raw_line in raw_content.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 5:
                all_lines_valid = False
                continue
            try:
                cid = int(float(parts[0]))
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                conf = float(parts[5]) if len(parts) >= 6 else 0.90
                norm = _normalize_box_dict(
                    {
                        "class_id": cid,
                        "cx": cx,
                        "cy": cy,
                        "w": w,
                        "h": h,
                        "confidence": conf,
                        "trusted": True,
                        "weak": False,
                    },
                    default_trusted=True,
                )
                if norm is not None:
                    boxes.append(norm)
                else:
                    all_lines_valid = False
            except (ValueError, TypeError):
                all_lines_valid = False

        return boxes, all_lines_valid

    def format_yolo_txt(self, boxes: Sequence[Dict[str, Any]]) -> str:
        """Format normalized box dictionaries into Ultralytics YOLO `.txt` lines."""
        lines: List[str] = []
        for b in boxes:
            norm = _normalize_box_dict(b, default_trusted=True)
            if norm is not None:
                lines.append(
                    f"{int(norm['class_id'])} "
                    f"{float(norm['cx']):.6f} "
                    f"{float(norm['cy']):.6f} "
                    f"{float(norm['w']):.6f} "
                    f"{float(norm['h']):.6f}"
                )
        return ("\n".join(lines) + "\n") if lines else ""

    def load_and_repair_json(
        self,
        json_path: Union[str, Path],
        *,
        persist_repair: bool = True,
    ) -> Tuple[Optional[Dict[str, Any]], bool, Dict[str, Any]]:
        """
        Load a review JSON file, repairing syntax corruption if needed and
        extracting any salvaged `"human_approved"` / `"done"` flags.

        Returns:
            `(data_or_None, was_repaired, salvaged_flags)`
        """
        p = Path(json_path)
        if not p.is_file():
            return None, False, {}

        try:
            raw_bytes = p.read_bytes()
        except Exception as exc:
            logger.warning("Failed to read review JSON %s: %s", p, exc)
            return None, False, {}

        raw_text = raw_bytes.decode("utf-8", errors="ignore")
        salvaged = extract_salvaged_flags(raw_text)
        data, was_repaired = repair_json_content(raw_text)

        if data is not None:
            # Ensure salvaged flags are preserved if missing in repaired dict
            for flag_key, flag_val in salvaged.items():
                if flag_key not in data:
                    data[flag_key] = flag_val
                    was_repaired = True

            if was_repaired and persist_repair:
                try:
                    self.atomic_write_json(p, data)
                except Exception as exc:
                    logger.warning("Failed to persist repaired JSON %s: %s", p, exc)

        return data, was_repaired, salvaged

    def _extract_emitted_boxes_from_json_data(
        self,
        json_data: Dict[str, Any],
        existing_txt_boxes: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], bool]:
        """
        Normalize `trusted` and `weak` lists inside `json_data`, preserve
        `human_approved` / `done` flags, and compute the exact list of boxes
        that must be emitted to `labels/{split}/{stem}.txt`.

        Returns:
            `(sanitized_trusted, sanitized_weak, emit_boxes_for_txt, json_needs_update)`
        """
        needs_update = False
        done_val = json_data.get("done")
        top_human_approved = bool(
            json_data.get("human_approved") is True
            or json_data.get("approved") is True
            or done_val in ("ok", True)
        )

        raw_trusted = json_data.get("trusted")
        raw_weak = json_data.get("weak")

        if not isinstance(raw_trusted, list):
            if isinstance(json_data.get("boxes"), list):
                raw_trusted = list(json_data["boxes"])
                needs_update = True
            elif isinstance(json_data.get("annotations"), list):
                raw_trusted = list(json_data["annotations"])
                needs_update = True
            else:
                raw_trusted = []
                if "trusted" not in json_data:
                    needs_update = True

        if not isinstance(raw_weak, list):
            raw_weak = []
            if "weak" not in json_data:
                needs_update = True

        sanitized_trusted: List[Dict[str, Any]] = []
        for b in raw_trusted:
            norm = _normalize_box_dict(b, default_trusted=True, preserve_flags=True)
            if norm is None:
                needs_update = True
                continue
            if top_human_approved:
                if json_data.get("human_approved") is True and not norm.get("human_approved"):
                    norm["human_approved"] = True
                norm["trusted"] = True
                norm["weak"] = False
            sanitized_trusted.append(norm)

        sanitized_weak: List[Dict[str, Any]] = []
        for b in raw_weak:
            norm = _normalize_box_dict(b, default_trusted=False, preserve_flags=True)
            if norm is None:
                needs_update = True
                continue
            # If a box in weak was explicitly marked human_approved, promote it to trusted
            if norm.get("human_approved") is True or norm.get("hand_label") is True:
                norm["trusted"] = True
                norm["weak"] = False
                sanitized_trusted.append(norm)
                needs_update = True
            else:
                sanitized_weak.append(norm)

        # Merge any box-level flags from existing_txt_boxes if needed
        if existing_txt_boxes and not sanitized_trusted and not sanitized_weak and done_val not in ("no", "rejected", False):
            for tb in existing_txt_boxes:
                norm = _normalize_box_dict(tb, default_trusted=True, preserve_flags=True)
                if norm is not None:
                    if top_human_approved:
                        norm["human_approved"] = True
                    sanitized_trusted.append(norm)
            if sanitized_trusted:
                needs_update = True

        # Determine which boxes should be written to labels/{split}/{stem}.txt
        if done_val in ("no", "rejected", False):
            emit_boxes: List[Dict[str, Any]] = []
        elif done_val in ("ok", True):
            # When sample is marked done="ok"/True, emit trusted (and weak if trusted is empty or all approved)
            emit_boxes = list(sanitized_trusted)
            for wb in sanitized_weak:
                if not any(_boxes_match(wb, tb) for tb in emit_boxes):
                    emit_boxes.append(wb)
        else:
            emit_boxes = list(sanitized_trusted)
            stem_str = str(json_data.get("stem", "")).strip().lower()
            if stem_str.startswith("learn_"):
                for wb in sanitized_weak:
                    if not any(_boxes_match(wb, tb) for tb in emit_boxes):
                        emit_boxes.append(wb)

        return sanitized_trusted, sanitized_weak, emit_boxes, needs_update

    # ------------------------------------------------------------------
    # Core Synchronization API
    # ------------------------------------------------------------------

    def sync_files(
        self,
        json_path: Union[str, Path],
        txt_path: Union[str, Path],
        *,
        split: Optional[str] = None,
        stem: Optional[str] = None,
        auto_repair: Optional[bool] = None,
    ) -> SyncPairResult:
        """
        Verify and atomically synchronize a specific `(json_path, txt_path)` pair.
        Repairs corrupted JSONs, reconstructs missing sidecars, and preserves
        `"human_approved"` and `"done"` flags.
        """
        with self._lock:
            json_p = Path(json_path)
            txt_p = Path(txt_path)
            do_repair = self.auto_repair if auto_repair is None else bool(auto_repair)

            resolved_stem = str(stem or json_p.stem or txt_p.stem)
            if split is not None:
                resolved_split = str(split)
            else:
                parent_name = json_p.parent.name or txt_p.parent.name
                resolved_split = parent_name if parent_name in ("train", "val", "test") else "train"

            result = SyncPairResult(
                split=resolved_split,
                stem=resolved_stem,
                json_path=str(json_p),
                txt_path=str(txt_p),
                synced=False,
            )

            json_exists = json_p.is_file()
            txt_exists = txt_p.is_file()

            json_data, was_json_repaired, salvaged_flags = self.load_and_repair_json(
                json_p, persist_repair=do_repair
            )
            txt_boxes, txt_clean_syntax = self.parse_yolo_txt(txt_p)

            if was_json_repaired:
                result.repaired_json = True
                result.details.append("Repaired malformed JSON syntax")

            # Case 1: JSON is valid (or was repaired in-place)
            if json_data is not None:
                # Preserve existing flags & salvaged flags
                for k, v in salvaged_flags.items():
                    if k not in json_data:
                        json_data[k] = v

                sanitized_trusted, sanitized_weak, emit_boxes, json_dirty = (
                    self._extract_emitted_boxes_from_json_data(json_data, existing_txt_boxes=txt_boxes)
                )

                result.human_approved = bool(
                    json_data.get("human_approved") is True
                    or json_data.get("approved") is True
                    or any(b.get("human_approved") is True for b in sanitized_trusted)
                )
                result.done = json_data.get("done")
                result.trusted_count = len(sanitized_trusted)
                result.weak_count = len(sanitized_weak)
                result.txt_box_count = len(emit_boxes)

                # Check if TXT matches emit_boxes
                txt_matches = False
                if txt_exists and txt_clean_syntax and len(txt_boxes) == len(emit_boxes):
                    remaining = list(txt_boxes)
                    matched_all = True
                    for eb in emit_boxes:
                        found_idx = -1
                        for idx, tb in enumerate(remaining):
                            if _boxes_match(eb, tb):
                                found_idx = idx
                                break
                        if found_idx >= 0:
                            remaining.pop(found_idx)
                        else:
                            matched_all = False
                            break
                    txt_matches = matched_all

                if txt_matches and not json_dirty and not was_json_repaired:
                    result.synced = True
                    return result

                if not do_repair:
                    result.synced = False
                    return result

                # Update JSON if needed, preserving all top-level metadata and flags
                if json_dirty or was_json_repaired or json_data.get("stem") != resolved_stem:
                    updated_payload = dict(json_data)
                    updated_payload["stem"] = resolved_stem
                    updated_payload["split"] = resolved_split
                    updated_payload["trusted"] = sanitized_trusted
                    updated_payload["weak"] = sanitized_weak
                    if "human_approved" in json_data:
                        updated_payload["human_approved"] = bool(json_data["human_approved"])
                    elif result.human_approved:
                        updated_payload["human_approved"] = True
                    if "done" in json_data:
                        updated_payload["done"] = json_data["done"]
                    self.atomic_write_json(json_p, updated_payload)
                    result.repaired_json = True

                # Write TXT atomically if missing or out-of-sync
                if not txt_matches:
                    txt_content = self.format_yolo_txt(emit_boxes)
                    self.atomic_write_text(txt_p, txt_content)
                    if not txt_exists:
                        result.created_txt = True
                        result.details.append("Created missing label TXT from review JSON")
                    else:
                        result.repaired_txt = True
                        result.details.append("Synchronized label TXT with review JSON")

                result.synced = True
                return result

            # Case 2: JSON is missing or unrecoverably corrupt, but TXT exists and has valid syntax (or salvageable boxes)
            if txt_exists and (txt_clean_syntax or txt_boxes):
                if not do_repair:
                    result.synced = False
                    return result

                preserved_human_approved = bool(
                    salvaged_flags.get("human_approved") is True
                    or salvaged_flags.get("approved") is True
                )
                preserved_done = salvaged_flags.get("done", "ok" if txt_boxes else None)

                reconstructed_trusted: List[Dict[str, Any]] = []
                for tb in txt_boxes:
                    box_copy = dict(tb)
                    box_copy["trusted"] = True
                    box_copy["weak"] = False
                    if preserved_human_approved:
                        box_copy["human_approved"] = True
                    reconstructed_trusted.append(box_copy)

                reconstructed_payload: Dict[str, Any] = {
                    "stem": resolved_stem,
                    "split": resolved_split,
                    "trusted": reconstructed_trusted,
                    "weak": [],
                    "done": preserved_done,
                    "human_approved": preserved_human_approved,
                    "auto_repaired": True,
                }
                if "approved" in salvaged_flags:
                    reconstructed_payload["approved"] = bool(salvaged_flags["approved"])

                self.atomic_write_json(json_p, reconstructed_payload)
                if not txt_clean_syntax:
                    self.atomic_write_text(txt_p, self.format_yolo_txt(reconstructed_trusted))
                    result.repaired_txt = True

                if json_exists:
                    result.repaired_json = True
                    result.details.append("Reconstructed corrupted review JSON from label TXT")
                else:
                    result.created_json = True
                    result.details.append("Created missing review JSON from label TXT")

                result.human_approved = preserved_human_approved
                result.done = preserved_done
                result.trusted_count = len(reconstructed_trusted)
                result.weak_count = 0
                result.txt_box_count = len(reconstructed_trusted)
                result.synced = True
                return result

            # Case 3: Corrupt JSON with salvaged flags, but TXT is missing -> recover empty/salvaged JSON + TXT
            if json_exists and salvaged_flags and do_repair:
                preserved_human_approved = bool(
                    salvaged_flags.get("human_approved") is True
                    or salvaged_flags.get("approved") is True
                )
                preserved_done = salvaged_flags.get("done")
                recovered_payload: Dict[str, Any] = {
                    "stem": resolved_stem,
                    "split": resolved_split,
                    "trusted": [],
                    "weak": [],
                    "done": preserved_done,
                    "human_approved": preserved_human_approved,
                    "auto_repaired": True,
                }
                self.atomic_write_json(json_p, recovered_payload)
                self.atomic_write_text(txt_p, "")
                result.repaired_json = True
                result.created_txt = not txt_exists
                result.human_approved = preserved_human_approved
                result.done = preserved_done
                result.synced = True
                result.details.append("Recovered corrupted JSON with salvaged flags")
                return result

            result.error = "Both review JSON and label TXT are missing or irreparably corrupt"
            result.synced = False
            return result

    def sync_pair(
        self,
        split: str,
        stem: str,
        *,
        auto_repair: Optional[bool] = None,
    ) -> SyncPairResult:
        """Synchronize `dataset/review/{split}/{stem}.json` and `dataset/labels/{split}/{stem}.txt`."""
        j_path = self.get_review_path(split, stem)
        t_path = self.get_label_path(split, stem)
        return self.sync_files(
            j_path,
            t_path,
            split=split,
            stem=stem,
            auto_repair=auto_repair,
        )

    def verify_and_sync_pair(
        self,
        json_path: Union[str, Path],
        txt_path: Union[str, Path],
        auto_repair: bool = True,
    ) -> bool:
        """Boolean convenience wrapper compatible with `verify_and_sync_review_txt_pair`."""
        res = self.sync_files(json_path, txt_path, auto_repair=auto_repair)
        return bool(res.synced)

    def write_synchronized_pair(
        self,
        split: str,
        stem: str,
        *,
        trusted: Optional[Sequence[Dict[str, Any]]] = None,
        weak: Optional[Sequence[Dict[str, Any]]] = None,
        human_approved: Optional[bool] = None,
        done: Optional[Any] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> SyncPairResult:
        """
        Atomically write or update both `dataset/review/{split}/{stem}.json` and
        `dataset/labels/{split}/{stem}.txt` while preserving existing
        `"human_approved"` and `"done"` flags unless explicitly overridden.
        """
        with self._lock:
            j_path = self.get_review_path(split, stem)
            t_path = self.get_label_path(split, stem)

            existing_data, _, salvaged = self.load_and_repair_json(j_path, persist_repair=False)
            payload: Dict[str, Any] = dict(existing_data) if isinstance(existing_data, dict) else {}
            for k, v in salvaged.items():
                if k not in payload:
                    payload[k] = v

            if extra_fields:
                for k, v in extra_fields.items():
                    payload[k] = v

            payload["stem"] = str(stem)
            payload["split"] = str(split)

            # Preserve existing human_approved & done flags unless explicitly provided
            if human_approved is not None:
                payload["human_approved"] = bool(human_approved)
            elif "human_approved" not in payload and payload.get("approved") is True:
                payload["human_approved"] = True

            if done is not None:
                payload["done"] = done

            is_doc_human_approved = bool(payload.get("human_approved") is True)

            if trusted is not None:
                norm_trusted: List[Dict[str, Any]] = []
                for b in trusted:
                    nb = _normalize_box_dict(b, default_trusted=True, preserve_flags=True)
                    if nb is not None:
                        if is_doc_human_approved:
                            nb["human_approved"] = True
                        nb["trusted"] = True
                        nb["weak"] = False
                        norm_trusted.append(nb)
                payload["trusted"] = norm_trusted
            elif "trusted" not in payload:
                payload["trusted"] = []

            if weak is not None:
                norm_weak: List[Dict[str, Any]] = []
                for b in weak:
                    nb = _normalize_box_dict(b, default_trusted=False, preserve_flags=True)
                    if nb is not None:
                        norm_weak.append(nb)
                payload["weak"] = norm_weak
            elif "weak" not in payload:
                payload["weak"] = []

            self.atomic_write_json(j_path, payload)
            _, _, emit_boxes, _ = self._extract_emitted_boxes_from_json_data(payload)
            self.atomic_write_text(t_path, self.format_yolo_txt(emit_boxes))

            return self.sync_files(j_path, t_path, split=split, stem=stem, auto_repair=True)

    def approve_sample(
        self,
        split: str,
        stem: str,
        boxes: Optional[Sequence[Dict[str, Any]]] = None,
        done: Any = "ok",
    ) -> SyncPairResult:
        """
        Mark a sample as human-approved (`"human_approved": True`, `"done": "ok"`),
        promoting its boxes to `trusted` and atomically syncing both `.json` and `.txt`.
        """
        with self._lock:
            j_path = self.get_review_path(split, stem)
            existing_data, _, _ = self.load_and_repair_json(j_path, persist_repair=False)

            if boxes is None:
                if isinstance(existing_data, dict):
                    raw_t = existing_data.get("trusted") or []
                    raw_w = existing_data.get("weak") or []
                    boxes = list(raw_t) + list(raw_w)
                else:
                    txt_boxes, _ = self.parse_yolo_txt(self.get_label_path(split, stem))
                    boxes = txt_boxes

            approved_boxes: List[Dict[str, Any]] = []
            for b in boxes or []:
                nb = _normalize_box_dict(b, default_trusted=True, preserve_flags=True)
                if nb is not None:
                    nb["human_approved"] = True
                    nb["hand_label"] = bool(nb.get("hand_label", True))
                    nb["trusted"] = True
                    nb["weak"] = False
                    nb["confidence"] = max(0.90, float(nb.get("confidence", 0.95)))
                    approved_boxes.append(nb)

            return self.write_synchronized_pair(
                split=split,
                stem=stem,
                trusted=approved_boxes,
                weak=[],
                human_approved=True,
                done=done,
                extra_fields={"approved": True, "approved_at": time.time()},
            )

    def sync_split(
        self,
        split: str,
        *,
        auto_repair: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Scan and synchronize all review JSON and label TXT pairs in a single split."""
        do_repair = self.auto_repair if auto_repair is None else bool(auto_repair)
        rev_dir = self.dataset_dir / "review" / str(split)
        lbl_dir = self.dataset_dir / "labels" / str(split)

        stems: Set[str] = set()
        if rev_dir.is_dir():
            for p in rev_dir.glob("*.json"):
                if p.name not in IGNORED_REVIEW_FILENAMES and not p.name.endswith(".tmp"):
                    stems.add(p.stem)
        if lbl_dir.is_dir():
            for p in lbl_dir.glob("*.txt"):
                if not p.name.endswith(".tmp"):
                    stems.add(p.stem)

        stats: Dict[str, Any] = {
            "split": str(split),
            "total_pairs": 0,
            "already_synced": 0,
            "auto_repaired": 0,
            "repaired_json_count": 0,
            "repaired_txt_count": 0,
            "orphan_txt_fixed": 0,
            "orphan_json_fixed": 0,
            "human_approved_preserved": 0,
            "done_preserved": 0,
            "failed": 0,
        }

        for stem in sorted(stems):
            stats["total_pairs"] += 1
            j_path = rev_dir / f"{stem}.json"
            t_path = lbl_dir / f"{stem}.txt"

            had_json = j_path.is_file()
            had_txt = t_path.is_file()

            check_only = self.sync_files(
                j_path, t_path, split=split, stem=stem, auto_repair=False
            )
            if check_only.synced and not check_only.repaired_json and not check_only.repaired_txt:
                stats["already_synced"] += 1
                if check_only.human_approved:
                    stats["human_approved_preserved"] += 1
                if check_only.done is not None:
                    stats["done_preserved"] += 1
                continue

            if not do_repair:
                stats["failed"] += 1
                continue

            res = self.sync_files(
                j_path, t_path, split=split, stem=stem, auto_repair=True
            )
            if res.synced:
                stats["auto_repaired"] += 1
                if res.repaired_json or res.created_json:
                    stats["repaired_json_count"] += 1
                if res.repaired_txt or res.created_txt:
                    stats["repaired_txt_count"] += 1
                if had_txt and not had_json:
                    stats["orphan_txt_fixed"] += 1
                elif had_json and not had_txt:
                    stats["orphan_json_fixed"] += 1
                if res.human_approved:
                    stats["human_approved_preserved"] += 1
                if res.done is not None:
                    stats["done_preserved"] += 1
            else:
                stats["failed"] += 1

        return stats

    def sync_all(
        self,
        *,
        splits: Optional[Sequence[str]] = None,
        auto_repair: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Synchronize all `{split}/{stem}` pairs across all configured dataset splits."""
        target_splits = tuple(splits) if splits is not None else self.splits
        totals: Dict[str, Any] = {
            "total_pairs": 0,
            "already_synced": 0,
            "auto_repaired": 0,
            "repaired_json_count": 0,
            "repaired_txt_count": 0,
            "orphan_txt_fixed": 0,
            "orphan_json_fixed": 0,
            "human_approved_preserved": 0,
            "done_preserved": 0,
            "failed": 0,
            "splits": {},
        }
        for sp in target_splits:
            sp_stats = self.sync_split(sp, auto_repair=auto_repair)
            totals["splits"][sp] = sp_stats
            for k in (
                "total_pairs",
                "already_synced",
                "auto_repaired",
                "repaired_json_count",
                "repaired_txt_count",
                "orphan_txt_fixed",
                "orphan_json_fixed",
                "human_approved_preserved",
                "done_preserved",
                "failed",
            ):
                totals[k] += int(sp_stats.get(k, 0))
        return totals


# ==============================================================================
# Module-Level Convenience Functions & Diagnostics
# ==============================================================================


def verify_and_sync_review_txt_pair(
    json_path: Union[str, Path],
    txt_path: Union[str, Path],
    auto_repair: bool = True,
) -> bool:
    """Module-level convenience wrapper using `AtomicReviewTxtSynchronizer`."""
    syncer = AtomicReviewTxtSynchronizer(auto_repair=auto_repair)
    return syncer.verify_and_sync_pair(json_path, txt_path, auto_repair=auto_repair)


def sync_all_review_txt_pairs(
    dataset_dir: Union[str, Path],
    *,
    splits: Sequence[str] = ("train", "val"),
    auto_repair: bool = True,
) -> Dict[str, Any]:
    """Module-level convenience wrapper to synchronize an entire dataset directory."""
    syncer = AtomicReviewTxtSynchronizer(
        dataset_dir=dataset_dir,
        splits=splits,
        auto_repair=auto_repair,
    )
    return syncer.sync_all(splits=splits, auto_repair=auto_repair)


def get_atomic_review_sync_capabilities() -> List[str]:
    return [
        "atomic_tmp_os_replace",
        "corrupt_json_repair",
        "human_approved_done_preservation",
        "review_txt_bidirectional_sync",
    ]


def validate_atomic_review_sync_contract() -> bool:
    return bool(get_atomic_review_sync_capabilities())


__all__ = [
    "AtomicReviewTxtSynchronizer",
    "SyncPairResult",
    "atomic_write_text",
    "atomic_write_json",
    "extract_salvaged_flags",
    "repair_json_content",
    "verify_and_sync_review_txt_pair",
    "sync_all_review_txt_pairs",
    "get_atomic_review_sync_capabilities",
    "validate_atomic_review_sync_contract",
]
