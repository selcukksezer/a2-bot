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

Status — F-1980 / Karar 6 (KALAN 225 düzeltme, dilim R62):
  * Bu modül canlıdır, silinmeyecek: public sync API'sini (`__all__`)
    sağlayan kanonik implementasyondur. Import eden test
    `tests/test_atomic_review_sync.py`'dir.
  * Çift implementasyon notu: `dataset_factory/dataset_validator.py`
    aynı yardımcıların (`atomic_write_text`, `atomic_write_json`,
    `verify_and_sync_review_txt_pair`, `sync_all_review_txt_pairs`)
    paralel bir kopyasını taşır; mevcut W41 testleri
    (`tests/test_w41_atomic_json_txt_sync.py`) o kopyayı çalıştırır,
    bu modülü değil. İki kopya birleştirilmedi — birleştirme kararı
    bu dilimin dışında bırakıldı.
  * Public dışa aktarma (`dataset_factory/__init__.py` üzerinden)
    R46 diliminde yapılır; bu modül `__init__.py`'ye dokunmaz.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from dataset_factory.approval_state import (
    box_has_human_provenance,
    is_frame_human_approved,
)

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

# P1-12 (K-31/K-34/K-35): güven sabitleri + K1/E-03/E-06 kapı predikatları
# tek kaynaktan (dataset_factory/label_trust.py) okunur. label_trust bu
# modülü import etmez (yalnızca dataset_validator), döngü yoktur; yine de
# bağımsız import senaryosuna karşı yedek tanımlar aşağıdadır.
try:
    from dataset_factory.label_trust import (
        TRAIN_CONF_FLOOR,
        annotation_confidence as _lt_annotation_confidence,
        has_measured_confidence as _lt_has_measured_confidence,
        is_learn_auto_annotation as _lt_is_learn_auto,
        _is_explicitly_human_approved as _lt_is_human,
        _is_hand_or_gather_seed as _lt_is_seed,
    )
except Exception:  # pragma: no cover - standalone import fallback
    TRAIN_CONF_FLOOR = 0.45

    def _lt_annotation_confidence(ann: Any) -> float:
        if not isinstance(ann, dict):
            return 0.0
        for _k in ("confidence", "conf", "score"):
            try:
                _v = float(ann[_k])
            except (KeyError, TypeError, ValueError):
                continue
            if _v == _v:
                return min(1.0, max(0.0, _v))
        return 0.0

    def _lt_has_measured_confidence(ann: Any) -> bool:
        if not isinstance(ann, dict):
            return False
        for _k in ("confidence", "conf", "score"):
            try:
                _v = float(ann[_k])
            except (KeyError, TypeError, ValueError):
                continue
            if _v == _v:
                return True
        return False

    def _lt_is_learn_auto(ann: Any) -> bool:
        if not isinstance(ann, dict):
            return False
        if ann.get("learn_auto") is True or ann.get("omni_synthetic") is True:
            return True
        return str(ann.get("source", "")).strip().lower().startswith("omni_")

    def _lt_is_human(ann: Any) -> bool:
        return bool(
            isinstance(ann, dict)
            and (ann.get("hand_label") is True or ann.get("human_approved") is True)
        )

    def _lt_is_seed(ann: Any) -> bool:
        if not isinstance(ann, dict):
            return False
        if ann.get("hand_label") is True:
            return True
        try:
            _cid = int(ann.get("class_id", -1))
        except (TypeError, ValueError):
            _cid = -1
        _src = str(ann.get("source", "")).lower()
        _key = str(ann.get("gather_key", ann.get("key", ""))).strip().lower()
        return bool(
            _cid == 13
            and (
                _key in ("f", "e")
                or "f_template" in _src
                or "e_template" in _src
                or "gather_prompt" in _src
            )
        )

DEFAULT_TRAIN_CONF_FLOOR: float = TRAIN_CONF_FLOOR
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

    When the text contains several occurrences of a flag (e.g. concatenated
    JSON objects `{...}{...}`), the LAST occurrence wins: it reflects the most
    recent write, while the first occurrence may be stale.
    """
    salvaged: Dict[str, Any] = {}
    if not raw_text:
        return salvaged

    def _last_match(pattern: re.Pattern[str]) -> Optional[re.Match[str]]:
        found: Optional[re.Match[str]] = None
        for m in pattern.finditer(raw_text):
            found = m
        return found

    m_ha = _last_match(_RE_HUMAN_APPROVED)
    if m_ha:
        salvaged["human_approved"] = m_ha.group(1).lower() == "true"

    m_app = _last_match(_RE_APPROVED)
    if m_app:
        salvaged["approved"] = m_app.group(1).lower() == "true"

    # "done" may appear as a string or as a bool/null; the last occurrence
    # of either form wins.
    last_done: Optional[Tuple[int, Any]] = None
    m_done_str = _last_match(_RE_DONE_STR)
    if m_done_str:
        last_done = (m_done_str.start(), m_done_str.group(1))
    m_done_bool = _last_match(_RE_DONE_BOOL)
    if m_done_bool:
        token = m_done_bool.group(1).lower()
        val: Any = True if token == "true" else (False if token == "false" else None)
        if last_done is None or m_done_bool.start() > last_done[0]:
            last_done = (m_done_bool.start(), val)
    if last_done is not None:
        salvaged["done"] = last_done[1]

    return salvaged


def _scan_outermost_object_spans(text: str) -> List[Tuple[int, int]]:
    """
    Legacy outermost-`{ ... }` span scanner (verbatim pre-D40 behaviour):
    brace counting with string/escape awareness. Used as the fallback when
    the `raw_decode` fast path bails out, so malformed input is handled
    exactly as before.
    """
    objects: List[Tuple[int, int]] = []
    brace_depth = 0
    start_idx = -1
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
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
                    objects.append((start_idx, idx + 1))
                    start_idx = -1

    return objects


def _extract_concatenated_object_spans(
    text: str,
) -> Optional[List[Tuple[int, int, Optional[Dict[str, Any]]]]]:
    """
    Fast path for step 2 of `repair_json_content`: find every top-level
    JSON object with the C `JSONDecoder.raw_decode` instead of a
    per-character Python scan, and reuse the already-parsed dicts so the
    merge step does not `json.loads` each object a second time.

    Returns a list of `(start, end, parsed_dict)` spans, or `None` when the
    fast path cannot prove it sees the same objects the legacy scanner
    would — the caller then falls back to `_scan_outermost_object_spans`
    (identical behaviour, just slower). Bail-outs:
      * a `"` in the gap before an opening `{`: the brace might sit inside
        a bare string, where only the legacy scanner's string tracking is
        trustworthy;
      * any `raw_decode` failure (malformed/truncated object, deep nesting):
        the legacy scanner + per-object comma-fix handles it as before.
    When it returns spans, they are exactly the legacy scanner's spans:
    every gap is quote-free and depth-0, and each span is a complete valid
    JSON object, so its end is the matching close brace; the parsed dict is
    exactly what `json.loads` of the span would return (same decoder).
    """
    decoder = json.JSONDecoder()
    objects: List[Tuple[int, int, Optional[Dict[str, Any]]]] = []
    search_from = 0
    while True:
        open_idx = text.find("{", search_from)
        if open_idx == -1:
            break
        if '"' in text[search_from:open_idx]:
            return None
        try:
            parsed, end_idx = decoder.raw_decode(text, open_idx)
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        objects.append((open_idx, end_idx, parsed))
        search_from = end_idx
    return objects


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

    # 2. Extract every balanced outermost JSON object '{ ... }' in the text.
    # Concatenated JSON (`{...}{...}`) must NOT silently drop the later
    # objects: parse each of them and merge them (list fields are extended,
    # scalar fields are taken from the latest object that sets them, and a
    # True approval flag in any object wins).
    # Fast path first (C raw_decode); on any ambiguity fall back to the
    # legacy char scanner, which handles malformed objects identically.
    objects = _extract_concatenated_object_spans(cleaned)
    if objects is None:
        objects = [
            (start, end, None)
            for (start, end) in _scan_outermost_object_spans(cleaned)
        ]

    if objects:
        merged: Dict[str, Any] = {}
        parsed_any = False
        for obj_start, obj_end, pre_parsed in objects:
            if pre_parsed is not None:
                # Fast path: raw_decode already parsed this exact span with
                # the same decoder json.loads would use — reuse it instead
                # of parsing (and regex-fixing) the span a second time.
                parsed_obj: Optional[Dict[str, Any]] = pre_parsed
            else:
                candidate = cleaned[obj_start:obj_end]
                # Remove trailing commas before } or ]
                candidate_fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                parsed_obj = None
                for attempt_str in (candidate, candidate_fixed):
                    try:
                        parsed = json.loads(attempt_str)
                        if isinstance(parsed, dict):
                            parsed_obj = parsed
                            break
                    except Exception:
                        pass
            if parsed_obj is None:
                continue
            parsed_any = True
            for key, value in parsed_obj.items():
                if (
                    key in ("trusted", "weak", "boxes", "annotations")
                    and isinstance(value, list)
                    and isinstance(merged.get(key), list)
                ):
                    # S154 (v19/P3-8): birleştirme dedupsuzdu — aynı kutu
                    # iki yapışık nesnede geçerse iki kez sayılıyordu.
                    # Birebir aynı öğe (JSON kanonik biçimiyle) bir kez
                    # tutulur; farklı öğeler aynen korunur.
                    merged_list = list(merged[key])
                    seen = set()
                    for _item in merged_list:
                        try:
                            seen.add(json.dumps(_item, sort_keys=True, default=str))
                        except Exception:
                            pass
                    for _item in value:
                        try:
                            _k = json.dumps(_item, sort_keys=True, default=str)
                        except Exception:
                            _k = None
                        if _k is None or _k not in seen:
                            merged_list.append(_item)
                            if _k is not None:
                                seen.add(_k)
                    merged[key] = merged_list
                elif key in ("human_approved", "approved"):
                    # A True approval in any object wins; a later False must
                    # not silently revoke an earlier True.
                    if value is True or key not in merged:
                        merged[key] = value
                else:
                    # Scalars: the latest object wins.
                    merged[key] = value
        if parsed_any:
            if len(objects) > 1:
                logger.warning(
                    "Repaired concatenated JSON: merged %d objects; "
                    "no object was dropped",
                    len(objects),
                )
            return merged, True

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
    `confidence`, `source`, `demoted_by_user`, and `class_name` when present.
    A box with `demoted_by_user=True` is never re-trusted by its origin
    flag (P0-8/K-12): demote is a newer explicit human decision.
    """
    if isinstance(raw, (list, tuple)) and len(raw) >= 5:
        try:
            cid = int(float(raw[0]))
            cx = float(raw[1])
            cy = float(raw[2])
            w = float(raw[3])
            h = float(raw[4])
            conf = float(raw[5]) if len(raw) >= 6 else None
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

    # A missing class id must NOT silently default to class 0 (herb_aria):
    # a box without an explicit class is invalid, not class 0.
    if not any(k in raw and raw[k] is not None for k in ("class_id", "cid", "cls")):
        return None

    try:
        cid = int(float(raw.get("class_id", raw.get("cid", raw.get("cls")))))
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
    # Out-of-range boxes are REJECTED, never silently clipped into range:
    # a box outside [0, 1] is corrupt input, not a box that happens to sit
    # on the frame edge.
    if cx < 0.0 or cx > 1.0 or cy < 0.0 or cy > 1.0 or w > 1.0 or h > 1.0:
        return None

    cx_c = round(cx, 6)
    cy_c = round(cy, 6)
    w_c = round(w, 6)
    h_c = round(h, 6)

    out: Dict[str, Any] = dict(raw) if preserve_flags else {}
    out["class_id"] = cid
    out["cx"] = cx_c
    out["cy"] = cy_c
    out["w"] = w_c
    out["h"] = h_c

    # Confidence is unknown when the source carries none: record None
    # instead of inventing 0.90 / 0.58.
    if "confidence" in raw and raw["confidence"] is not None:
        try:
            out["confidence"] = round(float(raw["confidence"]), 4)
        except (ValueError, TypeError):
            out["confidence"] = None
    else:
        out["confidence"] = None

    # Only an explicit per-box human decision counts as human approval
    # here: bare "approved"/"confirmed" are pipeline flags, not human
    # decisions (F-0828; sweep E-04). The flags are still preserved
    # verbatim above when present — only the trust inference narrows.
    # P0-8 (K-12): a user demote (``demoted_by_user=True``) is a NEWER,
    # explicit human decision than the origin flag ``hand_label`` — the
    # origin must survive demote, so the demote vetoes the hand→trusted
    # inference here. Clearing ``human_approved`` alone would NOT be
    # enough: this branch also reads ``hand_label``, which demote must
    # keep. A demoted box therefore keeps its caller-side trusted/weak
    # membership (weak) through persist instead of being rewritten
    # trusted into the JSON weak list.
    is_demoted = bool(out.get("demoted_by_user") is True)
    # P3-27: dar per-box insan-onayı okuması tek kanonik yoldan
    # (``approval_state.box_has_human_provenance``) — demote vetosu
    # korunur: demote edilmiş kutu kökeni ne olursa olsun trusted'a
    # dönmez (P0-8/K-12).
    is_human = box_has_human_provenance(out) and not is_demoted
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


def _may_promote_weak_box(box: Dict[str, Any]) -> bool:
    """Weak listesindeki kutu trusted'a terfi edebilir mi? (P3-26)

    Tek terfi predikatı: kutu düzeyinde açık insan kökeni
    (``hand_label is True`` veya per-box ``human_approved is True``)
    varsa terfi eder. Kullanıcı demote'u (``demoted_by_user is True``)
    her el-kökenli güven çıkarımını veto eder (P0-8/K-12): demote
    edilmiş kutu weak kalır — yoksa sync, persist'in bilerek weak
    bıraktığı kutuyu TXT'ye yazardı. ``_extract_emitted_boxes_from_json_data``
    ve ``write_synchronized_pair`` aynı predikatı okur.

    Köken kısmı kanonik ``approval_state.box_has_human_provenance``'a
    delege eder (P3-27 tek okuma); veto bu predikata özeldir.
    """
    if not isinstance(box, dict):
        return False
    if box.get("demoted_by_user") is True:
        return False
    return box_has_human_provenance(box)


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


def _txt_matches_emit(
    emit_boxes: Sequence[Dict[str, Any]],
    txt_boxes: Sequence[Dict[str, Any]],
    tol: float = 1e-3,
) -> bool:
    """
    Return True if `txt_boxes` contains exactly the same boxes as
    `emit_boxes` (same multiset, order-insensitive).

    Fast path: both sides are `_normalize_box_dict` outputs, whose
    coordinates are rounded to 6 decimals, and a TXT written by
    `format_yolo_txt` round-trips exactly at 6 decimals — so the common
    case (TXT generated by this module) matches by exact multiset keys in
    O(n) instead of the old O(n^2) greedy scan. Duplicates are handled by
    counting, not by set membership.

    Fallback: if any emit box has no exact key (e.g. a foreign TXT whose
    coordinates differ below `tol` but not exactly), the original
    tolerance-based greedy matching runs over the full lists, so the
    result is identical to the old behaviour in every case.
    """
    if len(emit_boxes) != len(txt_boxes):
        return False
    if not emit_boxes:
        return True

    def _exact_key(box: Dict[str, Any]) -> Tuple[int, float, float, float, float]:
        return (
            int(box["class_id"]),
            float(box["cx"]),
            float(box["cy"]),
            float(box["w"]),
            float(box["h"]),
        )

    try:
        remaining_counts: Dict[Tuple[int, float, float, float, float], int] = {}
        for tb in txt_boxes:
            key = _exact_key(tb)
            remaining_counts[key] = remaining_counts.get(key, 0) + 1
        for eb in emit_boxes:
            key = _exact_key(eb)
            count = remaining_counts.get(key, 0)
            if count == 0:
                break
            remaining_counts[key] = count - 1
        else:
            return True
    except (KeyError, TypeError, ValueError):
        pass

    # Tolerance fallback (foreign TXT files): verbatim old greedy matching.
    remaining = list(txt_boxes)
    for eb in emit_boxes:
        found_idx = -1
        for idx, tb in enumerate(remaining):
            if _boxes_match(eb, tb, tol=tol):
                found_idx = idx
                break
        if found_idx >= 0:
            remaining.pop(found_idx)
        else:
            return False
    return True


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
        auto_repair: bool = True,
    ) -> None:
        if dataset_dir is not None:
            self.dataset_dir = Path(dataset_dir).resolve()
        else:
            self.dataset_dir = Path(__file__).resolve().parent.parent / "dataset"

        self.splits: Tuple[str, ...] = tuple(str(s) for s in splits)
        # Confidence floor for training TXT emission. P3-26: the floor is
        # applied by the SINGLE emit rule, `train_txt_boxes` in
        # `dataset_factory.label_trust` (called from
        # `_extract_emitted_boxes_from_json_data`); a non-hand trusted box
        # below this floor never reaches the label TXT. Boxes with unknown
        # confidence (None — hand labels carry no score, S-6103) pass.
        self.floor: float = float(floor)
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
                # A YOLO TXT line carries no confidence and no provenance:
                # confidence is unknown (None) and the box is untrusted /
                # weak until a human approves it. Never invent 0.90/trusted.
                conf = float(parts[5]) if len(parts) >= 6 else None
                norm = _normalize_box_dict(
                    {
                        "class_id": cid,
                        "cx": cx,
                        "cy": cy,
                        "w": w,
                        "h": h,
                        "confidence": conf,
                        "trusted": False,
                        "weak": True,
                    },
                    default_trusted=False,
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
        stem: str = "",
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
        # done="ok" alone is NOT human approval, and neither is a bare
        # "approved" pipeline flag (F-0828; sweep E-04): only an explicit
        # human_approved counts as document-level approval. P3-27: okuma
        # tek kanonik yoldan (``approval_state.is_frame_human_approved``).
        top_human_approved = is_frame_human_approved(json_data)

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
                # Frame approval is provenance, not per-box inspection:
                # record bulk_approved, never stamp human_approved onto
                # boxes the human may not have looked at (sweep E-04).
                norm["bulk_approved"] = True
                norm["trusted"] = True
                norm["weak"] = False
            sanitized_trusted.append(norm)

        sanitized_weak: List[Dict[str, Any]] = []
        for b in raw_weak:
            norm = _normalize_box_dict(b, default_trusted=False, preserve_flags=True)
            if norm is None:
                needs_update = True
                continue
            # If a box in weak was explicitly marked human_approved, promote it to trusted.
            # P3-26: terfi tek predikattan (`_may_promote_weak_box`) —
            # kullanıcı demote'u terfiyi veto eder (P0-8/K-12): demote
            # edilmiş kutu weak kalır, yoksa sync onu TXT'ye yazardı.
            # (P3-27: predikatın köken kısmı kanonik
            # ``approval_state.box_has_human_provenance``'a delege eder.)
            if _may_promote_weak_box(norm):
                norm["trusted"] = True
                norm["weak"] = False
                sanitized_trusted.append(norm)
                needs_update = True
            else:
                sanitized_weak.append(norm)

        # Merge any box-level flags from existing_txt_boxes if needed.
        # TXT boxes carry no approval provenance: adopt them as WEAK unless
        # the document itself is explicitly human-approved, in which case
        # they are adopted as trusted. Never invent trust from a TXT file.
        if existing_txt_boxes and not sanitized_trusted and not sanitized_weak and done_val not in ("no", "rejected", False):
            for tb in existing_txt_boxes:
                norm = _normalize_box_dict(tb, default_trusted=False, preserve_flags=True)
                if norm is not None:
                    # Orphan TXT boxes prove nothing: adopt them as weak
                    # even when the document was approved — the approval
                    # was for the JSON's boxes, not for these (sweep E-04;
                    # same rule as sync_files Case 2).
                    norm["trusted"] = False
                    norm["weak"] = True
                    if top_human_approved:
                        norm["bulk_approved"] = True
                    sanitized_weak.append(norm)
            if sanitized_trusted or sanitized_weak:
                needs_update = True

        # Determine which boxes should be written to labels/{split}/{stem}.txt.
        # P3-26 (F026-02): the emit rule is SINGLE — `train_txt_boxes` in
        # `dataset_factory.label_trust` is the canonical rule, so this
        # synchronizer and `commit_split_labels` emit the same boxes. The
        # old local rule (trusted-only + `_passes_conf_floor`) emitted
        # below-floor human-approved weak boxes and fabricated-confidence
        # boxes that commit never writes. Weak (unreliable) boxes stay in
        # the review JSON for human review and are never emitted, whatever
        # done says — except the canonical rule's learn-stem approved-weak
        # branch. The done="no"/rejected gate stays here: a rejected frame
        # trains on zero boxes whatever the rule says.
        if done_val in ("no", "rejected", False):
            emit_boxes: List[Dict[str, Any]] = []
        else:
            from dataset_factory.label_trust import train_txt_boxes

            emit_stem = str(stem or json_data.get("stem") or "")
            emit_boxes = train_txt_boxes(
                sanitized_trusted, sanitized_weak, stem=emit_stem, floor=self.floor
            )

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

            # NOTE (D40): persist_repair=False on purpose — sync_files always
            # writes the final payload itself whenever a repair happened
            # (Case 1 writes the updated payload when was_json_repaired,
            # Cases 2/3 write the reconstructed/recovered payload), so the
            # intermediate persist inside load_and_repair_json was a
            # redundant atomic write (tmp + fsync + os.replace) per repaired
            # pair. Final on-disk state is identical; only the crash window
            # differs (a crash between load and the final write now leaves
            # the unrepaired file instead of the intermediately repaired one).
            json_data, was_json_repaired, salvaged_flags = self.load_and_repair_json(
                json_p, persist_repair=False
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
                    self._extract_emitted_boxes_from_json_data(
                        json_data, existing_txt_boxes=txt_boxes, stem=resolved_stem
                    )
                )

                # P1-12 (K-34/F080-K4): çıplak "approved" boru hattı
                # bayrağıdır — human_approved sayılmaz.
                result.human_approved = bool(
                    json_data.get("human_approved") is True
                    or any(b.get("human_approved") is True for b in sanitized_trusted)
                )
                result.done = json_data.get("done")
                result.trusted_count = len(sanitized_trusted)
                result.weak_count = len(sanitized_weak)
                result.txt_box_count = len(emit_boxes)

                # Check if TXT matches emit_boxes (order-insensitive;
                # _txt_matches_emit is O(n) on the exact-key fast path with
                # the old tolerance greedy scan as fallback).
                txt_matches = False
                if txt_exists and txt_clean_syntax:
                    txt_matches = _txt_matches_emit(emit_boxes, txt_boxes)

                # Fast path: only when the TXT already matches AND the
                # JSON stem/split agree with the resolved pair (S-6102).
                # A wrong-stem JSON must fall through to the repair
                # branch below instead of being reported synced as-is.
                if (
                    txt_matches
                    and not json_dirty
                    and not was_json_repaired
                    and json_data.get("stem") == resolved_stem
                    and json_data.get("split", resolved_split) == resolved_split
                ):
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
                # (P3-26: through `_rewrite_label_txt` — same emit rule,
                # .bak before overwrite; allow_empty=True because a repair
                # that empties the TXT is deliberate: nothing emittable
                # remains, so the old lines must go).
                if not txt_matches:
                    from dataset_factory.label_trust import _rewrite_label_txt

                    _rewrite_label_txt(txt_p, emit_boxes, allow_empty=True)
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

                # P1-12 (K-34/F080-K5): kurtarılan çıplak "approved"
                # bayrağı human_approved'a YÜKSELTİLMEZ — boru hattı
                # bayrağı olarak payload'da korunur, kutulara insan
                # onayı damgası basılmaz.
                preserved_human_approved = bool(
                    salvaged_flags.get("human_approved") is True
                )
                # An orphan TXT proves nothing about approval: done stays
                # None unless the salvaged flags actually say so, and the
                # boxes are reconstructed as WEAK, never trusted.
                preserved_done = salvaged_flags.get("done")

                reconstructed_weak: List[Dict[str, Any]] = []
                for tb in txt_boxes:
                    box_copy = dict(tb)
                    box_copy["trusted"] = False
                    box_copy["weak"] = True
                    # P0-1 (K-10): kurtarılan kutuya belge düzeyi onay
                    # damgası BASILMAZ — yetim TXT hiçbir onayı
                    # kanıtlamaz; kutu düzeyinde "human_approved"
                    # yalnızca açık insan kararıyla gelir. Damga
                    # basılsaydı bir sonraki sync'te
                    # `_may_promote_weak_box` bu kutuları trusted'a
                    # terfi ettirirdi. Belge düzeyi kurtarılmış onay
                    # yalnızca payload'ın üst düzeyinde korunur.
                    reconstructed_weak.append(box_copy)

                reconstructed_payload: Dict[str, Any] = {
                    "stem": resolved_stem,
                    "split": resolved_split,
                    "trusted": [],
                    "weak": reconstructed_weak,
                    "done": preserved_done,
                    "human_approved": preserved_human_approved,
                    "auto_repaired": True,
                }
                if "approved" in salvaged_flags:
                    reconstructed_payload["approved"] = bool(salvaged_flags["approved"])

                self.atomic_write_json(json_p, reconstructed_payload)
                # Only TRUSTED boxes may live in the training TXT: the
                # reconstructed boxes are weak (they survive in the review
                # JSON's weak list), so the TXT must be empty. Previously
                # the orphan TXT's weak boxes were left in place (or
                # re-written) while sync reported synced=True — and the
                # very next sync then emptied the TXT, i.e. the first sync
                # was not idempotent and untrusted boxes reached training.
                try:
                    _existing_txt = txt_p.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    _existing_txt = None
                if _existing_txt != "":
                    # P3-26: the clearing goes through `_rewrite_label_txt`
                    # (same writer, .bak before overwrite) instead of a
                    # second hand-rolled TXT write.
                    from dataset_factory.label_trust import _rewrite_label_txt

                    _rewrite_label_txt(txt_p, [], allow_empty=True)
                    result.repaired_txt = True
                    result.details.append(
                        "Cleared label TXT: reconstructed boxes are weak-only "
                        "(kept in review JSON)"
                    )

                if json_exists:
                    result.repaired_json = True
                    result.details.append("Reconstructed corrupted review JSON from label TXT")
                else:
                    result.created_json = True
                    result.details.append("Created missing review JSON from label TXT")

                result.human_approved = preserved_human_approved
                result.done = preserved_done
                result.trusted_count = 0
                result.weak_count = len(reconstructed_weak)
                result.txt_box_count = 0
                result.synced = True
                return result

            # Case 3: Corrupt JSON with salvaged flags, but TXT is missing -> recover empty/salvaged JSON + TXT
            if json_exists and salvaged_flags and do_repair:
                # P1-12 (K-34/F080-K5): çıplak "approved" human_approved'a yükseltilmez.
                preserved_human_approved = bool(
                    salvaged_flags.get("human_approved") is True
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
                # P3-26: same writer as the other TXT clears
                # (`_rewrite_label_txt`, .bak before overwrite).
                from dataset_factory.label_trust import _rewrite_label_txt

                _rewrite_label_txt(txt_p, [], allow_empty=True)
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

            # Preserve existing human_approved & done flags unless explicitly provided.
            # A bare "approved" pipeline flag must NOT become a human
            # approval (F-0828; sweep E-04).
            if human_approved is not None:
                payload["human_approved"] = bool(human_approved)

            if done is not None:
                payload["done"] = done

            is_doc_human_approved = is_frame_human_approved(payload)

            if trusted is not None:
                norm_trusted: List[Dict[str, Any]] = []
                for b in trusted:
                    nb = _normalize_box_dict(b, default_trusted=True, preserve_flags=True)
                    if nb is not None:
                        if is_doc_human_approved:
                            # Frame approval is provenance, not per-box
                            # inspection (sweep E-04): boxes passed in
                            # keep whatever per-box flag they already had.
                            nb["bulk_approved"] = True
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

            # P3-26: list membership follows the trust flag. A weak-listed
            # box with explicit per-box human provenance is trusted
            # (`_normalize_box_dict` already stamps trusted=True on it,
            # and `_extract_emitted_boxes_from_json_data` promotes it) —
            # if it stayed in the weak list, the pair writer would
            # evaluate it as weak while sync_files emits it as trusted
            # (F026-02). Demoted boxes never promote (P0-8/K-12).
            if isinstance(payload.get("trusted"), list) and isinstance(payload.get("weak"), list):
                _stay_weak: List[Dict[str, Any]] = []
                for _b in payload["weak"]:
                    if _may_promote_weak_box(_b):
                        _nb = dict(_b)
                        _nb["trusted"] = True
                        _nb["weak"] = False
                        payload["trusted"].append(_nb)
                    else:
                        _stay_weak.append(_b)
                payload["weak"] = _stay_weak

            # P3-26: the pair goes through the single writer in
            # label_trust (`write_review_pair`) — one emit rule
            # (`train_txt_boxes`), .bak before overwrite (P0-2), atomic
            # .tmp + os.replace for both files. allow_empty=True: persist
            # is a deliberate user write — the box list the user set is
            # the box list on disk, including an explicitly emptied frame
            # (otherwise a deleted box would silently come back from the
            # old TXT).
            from dataset_factory.label_trust import write_review_pair

            pair = write_review_pair(
                review_path=j_path,
                label_path=t_path,
                review_payload=payload,
                stem=str(stem),
                floor=self.floor,
                allow_empty=True,
            )
            sanitized_trusted, sanitized_weak, _extract_emit, _ = (
                self._extract_emitted_boxes_from_json_data(payload, stem=str(stem))
            )
            # The result reports what is actually on disk: the pair
            # writer's emit, not the extract's (they differ only for a
            # box the extract promotes — which the move above already
            # folded into the payload).
            emit_boxes = pair["emit_boxes"]

            # Do NOT call sync_files() again here: the pair was just written
            # from this payload, so re-syncing would redo the same work and
            # risk a second write. Build the result directly from the
            # payload that is already on disk.
            return SyncPairResult(
                split=str(split),
                stem=str(stem),
                json_path=str(j_path),
                txt_path=str(t_path),
                synced=True,
                human_approved=is_frame_human_approved(payload),
                done=payload.get("done"),
                trusted_count=len(sanitized_trusted),
                weak_count=len(sanitized_weak),
                txt_box_count=len(emit_boxes),
                details=["Wrote synchronized review JSON and label TXT"],
            )

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

        An explicit `boxes` list means the caller reviewed those boxes:
        they get per-box `human_approved`. A whole-frame approval
        (`boxes=None`) approves only boxes that were already trusted:
        weak boxes were never individually inspected, so they stay weak
        (carried over with a `bulk_approved` marker) instead of being
        promoted to human-approved (sweep E-04).
        """
        with self._lock:
            j_path = self.get_review_path(split, stem)
            existing_data, _, _ = self.load_and_repair_json(j_path, persist_repair=False)

            weak_carry: List[Dict[str, Any]] = []
            if boxes is None:
                if isinstance(existing_data, dict):
                    raw_t = existing_data.get("trusted") or []
                    raw_w = existing_data.get("weak") or []
                    boxes = list(raw_t)
                    weak_carry = list(raw_w)
                else:
                    txt_boxes, _ = self.parse_yolo_txt(self.get_label_path(split, stem))
                    boxes = txt_boxes

            approved_boxes: List[Dict[str, Any]] = []
            for b in boxes or []:
                nb = _normalize_box_dict(b, default_trusted=True, preserve_flags=True)
                if nb is not None:
                    # Explicit human approval: human_approved/trusted are
                    # the approval itself. Never invent hand_label (the box
                    # was not necessarily hand-drawn) and never inflate
                    # confidence to >= 0.90: the original score is kept.
                    nb["human_approved"] = True
                    nb["trusted"] = True
                    nb["weak"] = False
                    approved_boxes.append(nb)

            weak_boxes: List[Dict[str, Any]] = []
            for b in weak_carry:
                nb = _normalize_box_dict(b, default_trusted=False, preserve_flags=True)
                if nb is not None:
                    nb["bulk_approved"] = True
                    nb["trusted"] = False
                    nb["weak"] = True
                    weak_boxes.append(nb)

            return self.write_synchronized_pair(
                split=split,
                stem=stem,
                trusted=approved_boxes,
                weak=weak_boxes,
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

            # A single sync_files() call per pair: it already reports
            # whether the pair was already in sync (synced with no repair
            # flags), so a separate check-only pass would just do the same
            # work twice.
            res = self.sync_files(
                j_path, t_path, split=split, stem=stem, auto_repair=do_repair
            )
            if res.synced and not (
                res.repaired_json
                or res.repaired_txt
                or res.created_json
                or res.created_txt
            ):
                stats["already_synced"] += 1
                if res.human_approved:
                    stats["human_approved_preserved"] += 1
                if res.done is not None:
                    stats["done_preserved"] += 1
                continue

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
    auto_repair: bool = False,
) -> bool:
    """Module-level convenience wrapper using `AtomicReviewTxtSynchronizer`.

    P0-1 (K-10): ``auto_repair`` defaults to False, same as the
    ``dataset_validator`` copy — silent repair is opt-in only.
    """
    syncer = AtomicReviewTxtSynchronizer(auto_repair=auto_repair)
    return syncer.verify_and_sync_pair(json_path, txt_path, auto_repair=auto_repair)


def sync_all_review_txt_pairs(
    dataset_dir: Union[str, Path],
    *,
    splits: Sequence[str] = ("train", "val"),
    auto_repair: bool = False,
) -> Dict[str, Any]:
    """Module-level convenience wrapper to synchronize an entire dataset directory.

    P0-1 (K-10): ``auto_repair`` defaults to False (verify-only sweep).
    """
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
    """
    Actually exercise the sync contract (S-6101): never just report
    the hardcoded capability list as True.

    Runs a real round-trip in an isolated temporary dataset (nothing
    is written to the real `dataset/` tree):
      1. capabilities are non-empty,
      2. a human-approved pair can be written and re-verified synced,
      3. a wrong-stem JSON is repaired to the resolved stem instead
         of being reported synced as-is (S-6102),
      4. a low-confidence non-human trusted box does not reach the
         label TXT (confidence floor, S-6103).
    Returns True only if every check passes; any failure or
    exception returns False (and logs a warning).
    """
    try:
        if not get_atomic_review_sync_capabilities():
            return False
        with tempfile.TemporaryDirectory(prefix="atomic_review_sync_contract_") as tmp:
            syncer = AtomicReviewTxtSynchronizer(dataset_dir=tmp, splits=("train",))
            written = syncer.write_synchronized_pair(
                "train",
                "contract_check",
                trusted=[
                    {
                        "class_id": 0,
                        "cx": 0.5,
                        "cy": 0.5,
                        "w": 0.1,
                        "h": 0.1,
                        "confidence": 0.9,
                        "human_approved": True,
                    }
                ],
                weak=[],
                human_approved=True,
                done="ok",
            )
            if not written.synced:
                return False
            json_path = Path(tmp) / "review" / "train" / "contract_check.json"
            txt_path = Path(tmp) / "labels" / "train" / "contract_check.txt"
            if not json_path.is_file() or not txt_path.is_file():
                return False
            if not syncer.verify_and_sync_pair(json_path, txt_path):
                return False
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload["stem"] = "wrong_stem"
            atomic_write_json(json_path, payload)
            repaired = syncer.sync_files(json_path, txt_path)
            if not repaired.synced:
                return False
            fixed = json.loads(json_path.read_text(encoding="utf-8"))
            if fixed.get("stem") != "contract_check":
                return False
            low = syncer.write_synchronized_pair(
                "train",
                "contract_floor",
                trusted=[
                    {
                        "class_id": 0,
                        "cx": 0.5,
                        "cy": 0.5,
                        "w": 0.1,
                        "h": 0.1,
                        "confidence": 0.10,
                    }
                ],
                weak=[],
                human_approved=False,
                done="ok",
            )
            if not low.synced:
                return False
            low_txt = Path(tmp) / "labels" / "train" / "contract_floor.txt"
            if low_txt.read_text(encoding="utf-8").strip() != "":
                return False
        return True
    except Exception as exc:
        logger.warning("validate_atomic_review_sync_contract() failed: %s", exc)
        return False


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
