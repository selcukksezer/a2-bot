# -*- coding: utf-8 -*-
"""dataset_factory/approved_pool.py
===================================
Aion 2 Autonomous Dataset Factory - Approved Dataset Pool & Isolated Training Pipeline
Division 1 (A04, A07): Exclusive Isolated Pool for Human-Approved YOLO Training Samples.

User Requirement:
"onaylananlar kuyruktan eksilsin birde tekrar tekrar aç kapa yapınca geri gelmesin
onaylananlar ayrı bir yere gitsin eğitim için bekletilsin / onaylananlarla eğit seçeneği olsun hepsini eğitime almasın"

Key Capabilities:
1. Isolated Root Hierarchy (`dataset/approved/`):
   - `dataset/approved/images/train/`, `dataset/approved/images/val/`
   - `dataset/approved/labels/train/`, `dataset/approved/labels/val/`
   - `dataset/approved/approved_manifest.json`
   - `dataset/approved/data_approved.yaml`
2. Sample Quarantine & Dedup:
   - Once a sample is approved (`is_approved(stem) == True`), it never re-enters
     the Step 2 review queue on dialog open/close or disk rescan.
   - Atomically copies/saves image and verified YOLO `.txt` labels to `dataset/approved/`.
   - Records metadata, box counts, classes, and timestamps in `approved_manifest.json`.
3. Canonical 27-Class YOLO YAML Generation:
   - Generates valid Ultralytics `data_approved.yaml` pointing to approved images.
4. Selective Training Pipeline:
   - Enables "Sadece Onaylananlarla Eğit" (Train Only with Approved Samples).
   - Automatically supports 85%/15% train/val balanced distribution.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from dataset_factory.approval_state import (
    box_has_human_provenance,  # re-export: P1-10 test import'ları bu yoldan okur
    frame_has_provenance_box,
    is_frame_human_approved,
)

logger = logging.getLogger("DatasetFactory.ApprovedPool")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# =============================================================================
# Canonical Class Definitions Fallback (0..26: Core 22 + consumables + kinah + player)
# =============================================================================

DEFAULT_CANONICAL_NAMES: Dict[int, str] = {
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

try:
    from dataset_factory.class_taxonomy import CANONICAL_NAMES as _TAXONOMY_NAMES
    from dataset_factory.class_taxonomy import CORE_22_CANONICAL_NAMES
    CANONICAL_NAMES: Dict[int, str] = dict(_TAXONOMY_NAMES)
except Exception as exc:
    # S139 (v19/P3-8): yedek sessizdi — taksonomi import'u bozulursa
    # yerel kopya (eski 27 ad) fark edilmeden kullanılırdı. Artık
    # yüksek sesle loglanır; yedek hâlâ tam yerel kopyadır.
    logger.warning(
        "Taksonomi import edilemedi, yerel kanonik ad kopyası "
        "kullanılıyor (güncel olmayabilir): %s", exc
    )
    CORE_22_CANONICAL_NAMES = {k: v for k, v in DEFAULT_CANONICAL_NAMES.items() if k < 22}
    CANONICAL_NAMES = dict(DEFAULT_CANONICAL_NAMES)

_CANONICAL_IDS = frozenset(CANONICAL_NAMES.keys())


# =============================================================================
# Helper Utilities: Atomic Disk I/O & Bounding Box Normalization
# =============================================================================

def _fsync_file(path: Path) -> None:
    """Best-effort fsync so a tmp file's bytes reach disk before os.replace."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _safe_replace(src: Path, dst: Path, max_retries: int = 5, delay: float = 0.05) -> None:
    """Replace destination file atomically with retries for Windows file lock tolerance."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    for i in range(max_retries):
        try:
            os.replace(src, dst)
            return
        except OSError:
            if i == max_retries - 1:
                raise
            time.sleep(delay)


def _atomic_write_bytes(dest_path: Path, data: bytes) -> None:
    """Atomically write bytes to dest_path via a temp file + atomic rename."""
    tmp_path = dest_path.with_suffix(f"{dest_path.suffix}.tmp_{os.getpid()}_{time.time_ns()}")
    try:
        tmp_path.write_bytes(data)
        _fsync_file(tmp_path)
        _safe_replace(tmp_path, dest_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise


_YAML_PLAIN_SAFE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*")
_YAML_RESERVED_WORDS = frozenset(
    {"y", "n", "yes", "no", "true", "false", "on", "off", "null", "~"}
)


def _yaml_scalar(name: str) -> str:
    """Render one class name as a YAML scalar that round-trips via yaml.safe_load.

    Plain snake_case names stay unquoted (canonical output unchanged); anything
    else — ``: ``, ``#``, quotes, leading/trailing space, YAML 1.1 bool/null
    words (``yes``/``no``/``on``/…) or numeric-looking text — is emitted as a
    JSON double-quoted string, which is valid YAML and parses back exactly.
    """
    text = str(name)
    if (
        _YAML_PLAIN_SAFE.fullmatch(text)
        and text.lower() not in _YAML_RESERVED_WORDS
        and not text[0].isdigit()
    ):
        return text
    return json.dumps(text, ensure_ascii=False)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Atomically write text content using a temporary file and atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    try:
        tmp_path.write_text(text, encoding=encoding)
        _fsync_file(tmp_path)
        _safe_replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise


def atomic_copy_file(src_path: Path, dest_path: Path) -> None:
    """Atomically copy a file to dest_path using a temp file and atomic rename."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(f"{dest_path.suffix}.tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    try:
        shutil.copy2(src_path, tmp_path)
        _fsync_file(tmp_path)
        _safe_replace(tmp_path, dest_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise


# =============================================================================
# Image Header Validation (F-1451 / S-1903)
# =============================================================================
# Approved samples are stored as real image files; an arbitrary byte string
# must never be persisted as ".jpg"/".png". Magic-byte (SOI / PNG signature)
# checks reject corrupt payloads at ingest, before anything is written to
# the pool. This is intentionally a header check, not a full decode: a
# minimal SOI..EOI JPEG header is accepted, anything without the magic of
# its claimed format is refused.

_JPEG_SOI = b"\xff\xd8\xff"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_BMP_SIGNATURE = b"BM"
_WEBP_RIFF = b"RIFF"
_WEBP_SIGNATURE = b"WEBP"


def _image_magic_ok(data: bytes, suffix: str) -> bool:
    """True when `data` starts with the magic bytes of the image format for `suffix`."""
    if not data:
        return False
    suf = str(suffix or "").lower()
    if suf in (".jpg", ".jpeg"):
        return data[:3] == _JPEG_SOI
    if suf == ".png":
        return data[:8] == _PNG_SIGNATURE
    if suf == ".webp":
        return data[:4] == _WEBP_RIFF and data[8:12] == _WEBP_SIGNATURE
    if suf == ".bmp":
        return data[:2] == _BMP_SIGNATURE
    return False


def _image_file_magic_ok(path: Path) -> bool:
    """True when the file at `path` starts with the magic bytes of its suffix format."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(12)
    except OSError:
        return False
    return _image_magic_ok(header, path.suffix)


# Image formats the pool ever stores on disk (removal helpers delete any of
# these so a stale duplicate leaves nothing behind).
APPROVED_IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
# Image-like suffixes sync_from_disk deliberately does NOT register (B-11):
# the pool only accepts .jpg/.jpeg/.png (S-4601 — .webp/.bmp sources are
# transcoded to .jpg at ingest). The trainer counts .webp/.bmp/.tif/.tiff
# too, so a manually placed file of these formats would be counted by the
# trainer but missing from the manifest — sync surfaces the skip instead
# of silently ignoring it.
SKIPPED_IMAGE_EXTS: frozenset = frozenset({".webp", ".bmp", ".tif", ".tiff"})


def _normalize_box_with_reason(b: Any) -> Tuple[Optional[Tuple[int, float, float, float, float]], Optional[str]]:
    """
    Normalizes a single box representation into (class_id, cx, cy, w, h),
    plus the drop reason when it is rejected.

    Drop reasons (v19-pool-drop-count — "elenen sayılır" ilkesi; the
    reason vocabulary is asserted by
    tests/test_freeze_preserves_trusted_boxes.py):
      - "malformed": unparseable shape / missing fields / bad value type.
      - "zero_area": w <= 0 or h <= 0.
      - "non_canonical_class": class id outside the canonical 0..26 set.
      - "out_of_bounds": centre outside [0, 1] or w/h outside (0, 1].

    The rejection ORDER is load-bearing and unchanged from normalize_box:
    malformed -> zero_area -> non_canonical_class -> out_of_bounds.
    """
    try:
        if isinstance(b, str):
            parts = b.strip().split()
            if len(parts) < 5:
                return None, "malformed"
            cid = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        elif isinstance(b, dict):
            # A missing class id must NOT silently default to class 0.
            if not any(k in b and b[k] is not None for k in ("class_id", "cid", "cls", "label")):
                return None, "malformed"
            cid = int(float(b.get("class_id", b.get("cid", b.get("cls", b.get("label"))))))
            # Missing coordinates must not default to 0.0 / the centre.
            if not any(k in b and b[k] is not None for k in ("cx", "x_center", "x")):
                return None, "malformed"
            if not any(k in b and b[k] is not None for k in ("cy", "y_center", "y")):
                return None, "malformed"
            cx = float(b.get("cx", b.get("x_center", b.get("x"))))
            cy = float(b.get("cy", b.get("y_center", b.get("y"))))
            w = float(b.get("w", b.get("width", 0.0)))
            h = float(b.get("h", b.get("height", 0.0)))
        elif isinstance(b, (list, tuple)) and len(b) >= 5:
            cid = int(float(b[0]))
            cx = float(b[1])
            cy = float(b[2])
            w = float(b[3])
            h = float(b[4])
        else:
            return None, "malformed"

        if w <= 0.0 or h <= 0.0:
            return None, "zero_area"

        # F-0783: only canonical class ids (0..26) may enter the approved pool.
        if cid not in _CANONICAL_IDS:
            return None, "non_canonical_class"

        # Out-of-range boxes are REJECTED, never silently clipped into
        # range: a coordinate outside [0, 1] is corrupt input, not a box
        # that happens to sit on the frame edge.
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            return None, "out_of_bounds"

        return (cid, cx, cy, w, h), None
    except (ValueError, TypeError, KeyError):
        return None, "malformed"


def normalize_box(b: Any) -> Optional[Tuple[int, float, float, float, float]]:
    """
    Normalizes a single box representation into (class_id, cx, cy, w, h).
    Accepts dict, list, tuple, or YOLO-format string.

    Missing fields are REJECTED, never filled with template values
    (sweep E-05): a box without an explicit class is invalid, not
    class 0 (herb_aria); a box without explicit coordinates is invalid,
    not the frame centre; out-of-range coordinates are rejected, never
    silently clipped into range (same rule as
    atomic_review_sync._normalize_box_dict).

    Callers that need the rejection reason use
    ``_normalize_box_with_reason`` (same verdict, plus the reason).
    """
    norm, _reason = _normalize_box_with_reason(b)
    return norm


def parse_and_format_yolo_boxes(
    raw_boxes: Optional[Sequence[Any]],
    dropped_counts: Optional[Dict[str, int]] = None,
    *,
    stem: Optional[str] = None,
) -> Tuple[List[str], List[Tuple[int, float, float, float, float]], Set[int]]:
    """
    Parses arbitrary box collections into:
      1. List of YOLO lines: "cid cx cy w h"
      2. List of validated box tuples: (cid, cx, cy, w, h)
      3. Set of unique class IDs present.

    Behaviour is unchanged: a box that fails normalization is still
    NOT written to the pool TXT. What changed (v19-pool-drop-count):
    when ``dropped_counts`` is given, every dropped box is counted
    there by reason ("malformed" / "zero_area" /
    "non_canonical_class" / "out_of_bounds") and logged once
    (one warning line: stem + reason + box). Callers that pass no
    dict get exactly the old silent behaviour; the pool write paths
    (add_approved_sample, bulk_add_approved_samples) always pass one,
    so no box dropped on the pool TXT path leaves no trace.
    """
    lines: List[str] = []
    valid_boxes: List[Tuple[int, float, float, float, float]] = []
    classes_present: Set[int] = set()

    if not raw_boxes:
        return lines, valid_boxes, classes_present

    for item in raw_boxes:
        norm, reason = _normalize_box_with_reason(item)
        if norm is not None:
            cid, cx, cy, w, h = norm
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            valid_boxes.append(norm)
            classes_present.add(cid)
        elif dropped_counts is not None:
            dropped_counts[reason or "malformed"] = (
                dropped_counts.get(reason or "malformed", 0) + 1
            )
            logger.warning(
                "Havuz TXT yazımı: kutu elendi stem='%s' neden=%s kutu=%r",
                stem if stem else "?",
                reason or "malformed",
                item,
            )

    return lines, valid_boxes, classes_present


def _read_hand_label_txt(
    label_path: Path,
) -> Tuple[List[str], List[Tuple[int, float, float, float, float]], int]:
    """Read a labels TXT as hand-label evidence (faz-1 / P0-6 origin predicate).

    Returns ``(hand_lines, valid_boxes, model_lines_skipped)``:
      hand_lines: exactly-5-field lines. Only these can be hand labels —
        a 6+ field line is model output (the 6th field is a confidence
        score; dataset_factory/atomic_review_sync.py:714,
        dataset_factory/label_trust.py:1642) and is never hand-label
        evidence, so it is not even handed to the parser (normalize_box
        would otherwise accept a 6-field line on its first 5 fields).
      valid_boxes: ``parse_and_format_yolo_boxes(hand_lines)`` boxes.
      model_lines_skipped: number of non-5-field lines ignored.

    Raises OSError / UnicodeDecodeError when the file cannot be read;
    the caller counts that as a read error (never a silent skip).

    This is the single predicate shared by the freeze
    (``prepare_approved_training_dataset``, faz-1 7e43f9c + a359277) and
    the origin gate in ``sync_from_disk`` (P0-6 / K-23): freeze and sync
    must agree on what counts as a hand label, or freeze admits a frame
    that sync then drops (or sync admits what freeze rejects).

    NOTE (P0-1): the dialog-side canonical predicate is
    ``_reconcile_txt_hand_labels`` in ``dataset_factory/label_trust.py``
    (``_load_state``; parkta ``fix/yolo-etiket-txt-koruma`` dalında).
    It treats ANY parseable TXT line (5+ fields, 6-field model lines
    included) as a hand label, which is LOOSER than this 5-field-only
    freeze rule — so sync must NOT delegate to it as-is. If P0-1 lands
    a shared, importable predicate, this helper should delegate to it
    only if the exactly-5-field strictness is preserved. Format alone
    is weak provenance (ultra-tarama Ek B (a)): every TXT writer emits
    5-field lines, so the hand-label path ALSO requires that no review
    JSON exists for the stem (a reviewed frame is decided by its
    review JSON only — for the hand-label path; the pool-membership
    path, ``origin="pool"`` in ``sync_from_disk`` (P0-6 fixup), admits
    a reviewed-but-unapproved stem whose pool files + valid label
    prove it entered through an approval path).
    """
    raw_lines = [
        ln.strip()
        for ln in label_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    hand_lines = [ln for ln in raw_lines if len(ln.split()) == 5]
    _txt_lines, valid_boxes, _txt_classes = parse_and_format_yolo_boxes(hand_lines)
    return hand_lines, valid_boxes, len(raw_lines) - len(hand_lines)


def _is_hand_label_txt(label_path: Path) -> bool:
    """P0-6 origin predicate: True iff the TXT holds >=1 valid hand-label box.

    Unreadable / missing TXT is False (never an admission).
    """
    try:
        _hand_lines, valid_boxes, _skipped = _read_hand_label_txt(label_path)
    except Exception:
        return False
    return bool(valid_boxes)


def _review_approval_stems(
    review_dir: Path,
) -> Tuple[Set[str], Set[str], List[str]]:
    """Scan ``dataset/review/*.json`` for the sync origin gate (P0-6 / K-23).

    Returns ``(approved_stems, review_stems, parse_errors)``:
      approved_stems: stems whose review JSON has ``human_approved is
        True`` AND at least one trusted box with per-box human
        provenance (``hand_label`` / ``human_approved`` — K-21 / P1-10).
        F-0828: only an explicit human approval counts; ``done == "ok"``
        / ``bulk_approved`` / bare ``approved`` are pipeline flags, not
        human approval, and a frame-level flag alone admits nothing.
        P3-27: okuma tek kanonik yoldan
        (``approval_state.is_frame_human_approved`` +
        ``frame_has_provenance_box``); freeze kapısı
        (``prepare_approved_training_dataset``) AYNI fonksiyonları okur —
        iki kapı bir daha ayrı yazımla ayrışamaz.
      review_stems: every stem that has a review JSON at all (approved
        or not) — its presence closes the TXT-only path for that stem,
        exactly like the freeze does (faz-1): a reviewed frame is
        decided by its review JSON only.
      parse_errors: review files that could not be read/parsed.

    A stem is keyed by the JSON ``stem`` field when present, else the
    file stem; both keys are recorded (same rule as
    ``prepare_approved_training_dataset``).
    """
    approved_stems: Set[str] = set()
    review_stems: Set[str] = set()
    parse_errors: List[str] = []
    if not review_dir.is_dir():
        return approved_stems, review_stems, parse_errors
    for json_file in review_dir.rglob("*.json"):
        if json_file.name in ("trusted_seed.json", "manifest_stats.json"):
            continue
        review_stems.add(json_file.stem)
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            # Dosya adı ile `stem` alanı farklıysa ikisi de aynı kareye
            # işaret eder — review varlığı (onaylı olsun olmasın) TXT
            # yolunu iki anahtar için de kapatır.
            stem_field = payload.get("stem")
            if isinstance(stem_field, str) and stem_field.strip():
                review_stems.add(stem_field.strip())
            # K-21 / P1-10: kare bayrağı + en az bir köken kanıtlı kutu —
            # tek kanonik okuma (P3-27): is_frame_human_approved AND
            # frame_has_provenance_box. Bayrak tek başına havuza almaz.
            if is_frame_human_approved(payload) and frame_has_provenance_box(payload):
                approved_stems.add(json_file.stem)
                if isinstance(stem_field, str) and stem_field.strip():
                    approved_stems.add(stem_field.strip())
        except Exception as ex:
            parse_errors.append(str(json_file))
            logger.warning(
                "sync_from_disk: review JSON okunamadı, onay kanıtı sayılmadı %s: %s",
                json_file,
                ex,
            )
    return approved_stems, review_stems, parse_errors


# =============================================================================
# ApprovedDatasetPool Class
# =============================================================================

class ApprovedDatasetPool:
    """
    Manages the isolated `dataset/approved/` pool for human-approved frames.
    
    Guarantees:
    - Atomicity: Images and YOLO label files are written with temp-rename.
    - Dedup / Quarantine: Once a stem is marked approved, `is_approved(stem)`
      returns True so it never reappears in pending Step 2 queues.
    - YAML Export: Produces valid Ultralytics `data_approved.yaml` with
      canonical 27 classes.
    - Statistics: Real-time telemetry tracking total images, split counts,
      box counts, and class distributions.
    """

    def __init__(
        self,
        pool_root: Optional[Union[str, Path]] = None,
        project_root: Optional[Union[str, Path]] = None,
    ) -> None:
        if project_root is not None:
            p = Path(project_root).resolve()
            if p.name == "dataset":
                self.project_root = p.parent
                if pool_root is None:
                    pool_root = p / "approved"
            else:
                self.project_root = p
        else:
            self.project_root = Path(__file__).resolve().parent.parent

        if pool_root is not None:
            self.pool_root = Path(pool_root).resolve()
        else:
            self.pool_root = self.project_root / "dataset" / "approved"

        # Directory structure
        self.images_train_dir: Path = self.pool_root / "images" / "train"
        self.images_val_dir: Path = self.pool_root / "images" / "val"
        self.labels_train_dir: Path = self.pool_root / "labels" / "train"
        self.labels_val_dir: Path = self.pool_root / "labels" / "val"
        self.manifest_path: Path = self.pool_root / "approved_manifest.json"
        self.data_yaml_path: Path = self.pool_root / "data_approved.yaml"

        self._lock = threading.RLock()
        self._approved_stems: Set[str] = set()
        # v19-pool-drop-count: son bulk_add_approved_samples çağrısında
        # havuz TXT'sine yazılmadan elenen kutular, neden bazında
        # (bkz. _normalize_box_with_reason). bulk_add int döndürmeye
        # devam eder (geriye uyumluluk); sayaç buradan ve dondurma
        # sonucundaki dropped_pool_boxes* alanlarından okunur.
        self.last_bulk_dropped_boxes: Dict[str, int] = {}
        self._manifest: Dict[str, Any] = {
            "version": "1.0",
            "updated_at": 0.0,
            "total_approved_images": 0,
            "train_count": 0,
            "val_count": 0,
            "total_boxes": 0,
            "classes": {},
            "stems": {},
        }

        # Initialize directory layout and reload manifest
        self.ensure_pool_structure()
        self._load_or_rebuild_manifest()

    # -------------------------------------------------------------------------
    # Structure & Initialization
    # -------------------------------------------------------------------------

    def ensure_pool_structure(self) -> None:
        """Create the complete `dataset/approved/` directory hierarchy."""
        with self._lock:
            self.images_train_dir.mkdir(parents=True, exist_ok=True)
            self.images_val_dir.mkdir(parents=True, exist_ok=True)
            self.labels_train_dir.mkdir(parents=True, exist_ok=True)
            self.labels_val_dir.mkdir(parents=True, exist_ok=True)

    def _load_or_rebuild_manifest(self) -> None:
        """Load `approved_manifest.json` from disk, or rebuild from disk files if missing/corrupt."""
        with self._lock:
            loaded_ok = False
            if self.manifest_path.is_file():
                try:
                    data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and "stems" in data:
                        self._manifest = data
                        self._approved_stems = set(data.get("stems", {}).keys())
                        loaded_ok = True
                except Exception as ex:
                    logger.warning("Manifest parse failed, backing up and rebuilding: %s", ex)
                    backup = self.manifest_path.with_suffix(".corrupt.json")
                    try:
                        shutil.copy2(self.manifest_path, backup)
                    except Exception:
                        pass

            if not loaded_ok:
                self.sync_from_disk()
            elif self._recalculate_manifest_stats():
                # Manifest parsed but its stored counters disagreed with its
                # own stems (crash between stem write and stat update, hand
                # edit, older writer) — persist the repaired counters so
                # get_approved_stats() never reports stale numbers again.
                manifest_json = json.dumps(self._manifest, indent=2, ensure_ascii=False)
                self._backup_manifest_before_write()
                atomic_write_text(self.manifest_path, manifest_json, encoding="utf-8")

    # -------------------------------------------------------------------------
    # Split Directories Helper
    # -------------------------------------------------------------------------

    def images_dir_for_split(self, split: str) -> Path:
        """Return image directory for a split ('train' or 'val')."""
        return self.images_val_dir if str(split).lower().strip() == "val" else self.images_train_dir

    def labels_dir_for_split(self, split: str) -> Path:
        """Return label directory for a split ('train' or 'val')."""
        return self.labels_val_dir if str(split).lower().strip() == "val" else self.labels_train_dir

    @staticmethod
    def _resolve_split(clean_stem: str, split: str, val_ratio: float = 0.15) -> str:
        """Resolve 'train'/'val'; anything else is deterministically hashed
        (MD5 of the stem) to a train/val assignment honouring ``val_ratio``,
        so the same stem always lands in the same split no matter which entry
        point (single add, bulk add, rebalance) resolved it — adding or
        removing one frame never moves another frame's split (F070/K-24).

        The hash is per-stem only: ``h % 1000 < round(val_ratio * 1000)``
        (per-mille granularity, so ratios such as 0.16 from the 84% train
        slider are honoured instead of silently collapsing to 85/15)."""
        norm_split = str(split).lower().strip()
        if norm_split in ("train", "val"):
            return norm_split
        ratio = min(0.9, max(0.0, float(val_ratio)))
        h = int(hashlib.md5(clean_stem.encode("utf-8")).hexdigest(), 16)
        return "val" if (h % 1000) < int(round(ratio * 1000)) else "train"

    def _delete_sample_files(
        self, clean_stem: str, splits: Sequence[str], image_exts: Sequence[str]
    ) -> bool:
        """Delete a stem's image + label files for the given splits.

        Returns True if at least one file was removed. Shared by
        ``remove_sample`` and ``remove_approved_sample`` (both delete in
        both splits, so a re-approved stem leaves no orphan for
        sync_from_disk to resurrect); the two methods keep their own
        return semantics, only the unlink loops are shared.
        """
        removed_any_file = False
        for split in splits:
            img_dir = self.images_dir_for_split(split)
            lbl_dir = self.labels_dir_for_split(split)
            for ext in image_exts:
                p = img_dir / f"{clean_stem}{ext}"
                try:
                    if p.is_file():
                        p.unlink()
                        removed_any_file = True
                except OSError:
                    pass
            lbl = lbl_dir / f"{clean_stem}.txt"
            try:
                if lbl.is_file():
                    lbl.unlink()
                    removed_any_file = True
            except OSError:
                pass
        return removed_any_file

    # -------------------------------------------------------------------------
    # Core Operation: Add Approved Sample
    # -------------------------------------------------------------------------

    def add_approved_sample(
        self,
        stem: str,
        split: str = "train",
        image_path: Optional[Union[str, Path, Any]] = None,
        trusted_boxes: Optional[Sequence[Any]] = None,
        image_bytes: Optional[bytes] = None,
        approved_at: Optional[float] = None,
        boxes: Optional[Sequence[Any]] = None,
        val_ratio: float = 0.15,
    ) -> bool:
        """
        Atomically copies/saves image and verified YOLO `.txt` label into the approved pool,
        and registers the stem in `approved_manifest.json`.

        Args:
            stem: Clean image stem identifier (e.g. 'frame_000123').
            split: 'train' or 'val'. If other/empty, deterministically assigns
                by stem hash honouring `val_ratio` (see `_resolve_split`).
            image_path: Path to source image file, or numpy array. If None, resolves from dataset.
            trusted_boxes: List of trusted bounding boxes (dicts, tuples, or YOLO lines).
            image_bytes: Optional raw image bytes if already loaded in memory.
            approved_at: Optional approval epoch timestamp. Defaults to time.time().
            boxes: Optional alias for trusted_boxes to support robust callers.
            val_ratio: Val share used only by the hash fallback (default 0.15).

        Returns:
            bool: True if sample was successfully persisted and recorded; False otherwise.
        """
        if trusted_boxes is None and boxes is not None:
            trusted_boxes = boxes

        if not stem:
            return False

        clean_stem = Path(stem).stem.strip()
        if not clean_stem:
            return False

        norm_split = self._resolve_split(clean_stem, split, val_ratio=val_ratio)

        timestamp = float(approved_at) if approved_at is not None else time.time()

        # F-0784: validate boxes BEFORE touching disk — a zero-box sample is
        # not approved data and must never enter the pool with an empty label.
        # v19-pool-drop-count: elenen kutular sayılır + loglanır (havuz
        # TXT'sine yazılmayan kutunun izi kalır); davranış değişmez.
        _sample_dropped: Dict[str, int] = {}
        lines, valid_boxes, classes_present = parse_and_format_yolo_boxes(
            trusted_boxes, _sample_dropped, stem=clean_stem
        )
        if not valid_boxes:
            logger.error("Refusing zero-box approved sample '%s'", clean_stem)
            return False

        # Disk I/O happens OUTSIDE the lock: image copy/encode + label write
        # used to hold the pool lock, blocking is_approved() (the hot
        # quarantine check on the review dialog) for the whole I/O. Only the
        # manifest commit below needs the lock. Files written here but not
        # yet committed are re-registered by sync_from_disk() (they carry
        # valid labels), so a crash between the two phases self-heals.
        dest_images_dir = self.images_dir_for_split(norm_split)
        dest_labels_dir = self.labels_dir_for_split(norm_split)
        try:
            resolved_img_path, dest_img_name = self._save_or_copy_image(
                clean_stem=clean_stem,
                norm_split=norm_split,
                image_path=image_path,
                image_bytes=image_bytes,
                dest_images_dir=dest_images_dir,
            )
            if resolved_img_path is None and dest_img_name is None:
                logger.error("Could not find or save image for approved stem '%s'", clean_stem)
                return False

            # 2. Write the already-validated YOLO label (non-empty by construction)
            txt_content = "\n".join(lines) + ("\n" if lines else "")
            dest_label_path = dest_labels_dir / f"{clean_stem}.txt"
            atomic_write_text(dest_label_path, txt_content, encoding="utf-8")

            # 2b. Drop stale files if this stem was previously approved
            # into the other split (new image+label are on disk now).
            self._remove_other_split_files(clean_stem, norm_split)
        except Exception as ex:
            logger.exception("Failed to add approved sample '%s': %s", clean_stem, ex)
            return False

        # 3. Commit: manifest + in-memory index, under the lock only.
        with self._lock:
            # A concurrent remove may have deleted the files just written;
            # never commit a manifest entry pointing at missing files.
            if not (dest_images_dir / str(dest_img_name)).is_file() or not dest_label_path.is_file():
                logger.error("Image/label for approved stem '%s' vanished before commit", clean_stem)
                return False
            try:
                stems_map = self._manifest.setdefault("stems", {})
                stems_map[clean_stem] = {
                    "stem": clean_stem,
                    "split": norm_split,
                    "image_file": dest_img_name,
                    "label_file": f"{clean_stem}.txt",
                    "box_count": len(valid_boxes),
                    "classes": sorted(list(classes_present)),
                    "approved_at": timestamp,
                    "source_image": str(resolved_img_path) if resolved_img_path else "",
                }

                self._approved_stems.add(clean_stem)
                self._recalculate_manifest_stats_and_save()
                return True

            except Exception as ex:
                logger.exception("Failed to add approved sample '%s': %s", clean_stem, ex)
                return False

    def remove_sample(self, stem: str) -> bool:
        """Remove an approved sample from the pool and manifest."""
        clean_stem = str(stem or "").strip()
        if not clean_stem:
            return False
        with self._lock:
            stems_map = self._manifest.setdefault("stems", {})
            entry = stems_map.pop(clean_stem, None)
            self._approved_stems.discard(clean_stem)
            removed_any_file = self._delete_sample_files(
                clean_stem,
                ("train", "val"),
                (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
            )
            # F-0913: report False when nothing was actually removed.
            if entry is None and not removed_any_file:
                return False
            self._recalculate_manifest_stats_and_save()
            return True

    def _remove_other_split_files(self, clean_stem: str, keep_split: str) -> None:
        """Delete image+label of `clean_stem` in every split except `keep_split`.

        A stem re-approved into another split must not leave its old files
        behind: the same image in both train/ and val/ is train/val leakage,
        and sync_from_disk would resurrect the stale copy. Called only AFTER
        the new split's image+label are safely on disk, so a failed save
        never destroys the previously approved copy.
        """
        for split, img_dir, lbl_dir in (
            ("train", self.images_train_dir, self.labels_train_dir),
            ("val", self.images_val_dir, self.labels_val_dir),
        ):
            if split == keep_split:
                continue
            for ext in APPROVED_IMAGE_EXTS:
                try:
                    p = img_dir / f"{clean_stem}{ext}"
                    if p.is_file():
                        p.unlink()
                except OSError:
                    pass
            try:
                p = lbl_dir / f"{clean_stem}.txt"
                if p.is_file():
                    p.unlink()
            except OSError:
                pass

    def _save_or_copy_image(
        self,
        clean_stem: str,
        norm_split: str,
        image_path: Optional[Union[str, Path, Any]],
        image_bytes: Optional[bytes],
        dest_images_dir: Path,
    ) -> Tuple[Optional[Path], Optional[str]]:
        """Internal helper to copy or write image to destination directory atomically."""
        # 1. Direct bytes provided
        if image_bytes is not None and len(image_bytes) > 0:
            # F-1451/S-1903: bytes stored as .jpg must carry a JPEG SOI
            # magic — an arbitrary byte string is not an image and is
            # refused before anything is written to the pool.
            if not _image_magic_ok(image_bytes, ".jpg"):
                logger.error(
                    "Refusing non-JPEG image bytes for approved stem '%s' "
                    "(missing JPEG SOI magic)",
                    clean_stem,
                )
                return None, None
            dest_name = f"{clean_stem}.jpg"
            dest_path = dest_images_dir / dest_name
            _atomic_write_bytes(dest_path, image_bytes)
            return dest_path, dest_name

        # 2. Check if destination image already exists
        # S-4602: an existing dest file is only reused when its header
        # matches its suffix magic. A corrupt/stale dest file must not
        # win over a valid image_path/image_bytes — fall through so the
        # source is (re-)copied over it. (.webp/.bmp dest files are
        # deliberately not reused: S-4601, sync_from_disk only accepts
        # .jpg/.png/.jpeg, so they must be transcoded from source.)
        for ext in (".jpg", ".png", ".jpeg"):
            cand = dest_images_dir / f"{clean_stem}{ext}"
            if cand.is_file() and _image_file_magic_ok(cand):
                return cand, cand.name

        # 3. Handle numpy image array
        if image_path is not None and hasattr(image_path, "shape") and hasattr(image_path, "dtype"):
            try:
                import cv2
                dest_name = f"{clean_stem}.jpg"
                dest_path = dest_images_dir / dest_name
                success, buf = cv2.imencode(".jpg", image_path)
                if success:
                    _atomic_write_bytes(dest_path, buf.tobytes())
                    return dest_path, dest_name
            except Exception as ex:
                logger.warning("Failed to save numpy array for stem '%s': %s", clean_stem, ex)

        # 4. Resolve from filepath
        src_candidate: Optional[Path] = None
        if image_path is not None and isinstance(image_path, (str, Path)):
            p = Path(image_path)
            if p.is_file():
                src_candidate = p
            elif (self.project_root / p).is_file():
                src_candidate = (self.project_root / p).resolve()

        if src_candidate is None:
            # Fallback search in standard dataset image directories
            search_dirs = [
                self.project_root / "dataset" / "images" / norm_split,
                self.project_root / "dataset" / "images" / "train",
                self.project_root / "dataset" / "images" / "val",
                self.project_root / "dataset" / "images",
            ]
            for sdir in search_dirs:
                if not sdir.is_dir():
                    continue
                for ext in (".jpg", ".png", ".jpeg"):
                    cand = sdir / f"{clean_stem}{ext}"
                    if cand.is_file():
                        src_candidate = cand
                        break
                if src_candidate is not None:
                    break

        if src_candidate is not None and src_candidate.is_file():
            ext = src_candidate.suffix.lower()
            # F-1451/S-1903: a matching suffix alone does not make a file an
            # image — validate the source header before copying it in.
            if not _image_file_magic_ok(src_candidate):
                logger.error(
                    "Refusing to copy non-image source '%s' for approved "
                    "stem '%s' (header does not match %s magic)",
                    src_candidate,
                    clean_stem,
                    ext,
                )
                return None, None
            if ext in (".webp", ".bmp"):
                # S-4601: copying WEBP/BMP bytes under a ".jpg" name leaves
                # a content/extension mismatch that sync_from_disk drops
                # on reload (it only accepts .jpg/.png/.jpeg). Transcode
                # to JPEG so the stored file matches its extension, and
                # refuse instead of writing mismatched bytes if the
                # conversion fails.
                dest_name = f"{clean_stem}.jpg"
                dest_path = dest_images_dir / dest_name
                try:
                    import cv2

                    img = cv2.imread(str(src_candidate), cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError("cv2 could not decode source image")
                    success, buf = cv2.imencode(".jpg", img)
                    if not success:
                        raise ValueError("cv2 JPEG encode failed")
                except Exception as ex:
                    logger.error(
                        "Refusing to copy '%s' for approved stem '%s': "
                        "WEBP/BMP to JPEG conversion failed (%s)",
                        src_candidate,
                        clean_stem,
                        ex,
                    )
                    return None, None
                tmp_path = dest_path.with_suffix(f"{dest_path.suffix}.tmp_{os.getpid()}_{time.time_ns()}")
                tmp_path.write_bytes(buf.tobytes())
                _safe_replace(tmp_path, dest_path)
                return src_candidate, dest_name
            if ext not in (".png", ".jpg", ".jpeg"):
                # Defensive: _image_file_magic_ok only passes for the five
                # formats handled above, so this is unreachable — refuse
                # rather than writing bytes under a wrong extension.
                logger.error(
                    "Refusing to copy source '%s' for approved stem '%s' "
                    "(unsupported image extension %s)",
                    src_candidate,
                    clean_stem,
                    ext,
                )
                return None, None
            dest_name = f"{clean_stem}{ext}"
            dest_path = dest_images_dir / dest_name
            atomic_copy_file(src_candidate, dest_path)
            return src_candidate, dest_name

        return None, None

    # -------------------------------------------------------------------------
    # Bulk Operations
    # -------------------------------------------------------------------------

    def bulk_add_approved_samples(
        self,
        samples: Sequence[Dict[str, Any]],
        val_ratio: float = 0.15,
    ) -> int:
        """
        Adds multiple approved samples efficiently, updating the manifest once at the end.

        Each item in `samples` should be a dict with:
          - stem (required)
          - split (optional, default 'train')
          - image_path (optional)
          - trusted_boxes (optional)
          - approved_at (optional)

        `val_ratio` is honoured only by the hash fallback in `_resolve_split`
        (explicit 'train'/'val' splits are kept as-is).

        Returns the number of samples committed to the manifest (int,
        unchanged). Boxes dropped by normalization on the pool TXT path
        (v19-pool-drop-count) are counted by reason in
        ``self.last_bulk_dropped_boxes`` — the int return never carried
        them, so the counter lives on the instance (and, for freeze,
        in the prepare stats as ``dropped_pool_boxes*``).
        """
        # v19-pool-drop-count: her çağrı kendi sayacıyla başlar; boş
        # girdi bile önceki turun sayacını geride bırakmaz.
        self.last_bulk_dropped_boxes = {}
        if not samples:
            return 0

        # Phase 1 — disk I/O WITHOUT the lock: parse boxes, copy images and
        # write labels per sample. Holding the pool lock across a whole bulk
        # used to block is_approved() (the hot quarantine check) for the
        # entire batch; per-sample files are independent, so no lock is
        # needed until the manifest commit below.
        pending: List[Dict[str, Any]] = []
        dropped_total: Dict[str, int] = {}
        for s in samples:
            if not isinstance(s, dict):
                logger.warning("Skipping non-dict bulk sample: %r", type(s).__name__)
                continue
            stem = s.get("stem")
            if not stem or not isinstance(stem, (str, Path)):
                continue
            clean_stem = Path(stem).stem.strip()
            norm_split = self._resolve_split(clean_stem, s.get("split", "train"), val_ratio=val_ratio)

            # F-1030: honour the `boxes` alias too, like add_approved_sample.
            # Parse before touching disk so a zero-box sample leaves no orphan image.
            # v19-pool-drop-count: elenen kutu sayılır + loglanır — sıfır
            # kutulu atlanan örnekte bile elenen kutular sayaca girer.
            sample_dropped: Dict[str, int] = {}
            lines, valid_boxes, classes_present = parse_and_format_yolo_boxes(
                s.get("trusted_boxes") or s.get("trusted") or s.get("boxes"),
                sample_dropped,
                stem=clean_stem,
            )
            for reason, count in sample_dropped.items():
                dropped_total[reason] = dropped_total.get(reason, 0) + count
            # F-0784: zero-box samples are not approved data — skip, do not count.
            if not valid_boxes:
                logger.warning("Skipping zero-box bulk sample '%s'", clean_stem)
                continue

            dest_images_dir = self.images_dir_for_split(norm_split)
            dest_labels_dir = self.labels_dir_for_split(norm_split)

            # One bad sample must not abort the whole batch (previously
            # an exception here propagated, left earlier samples' files
            # on disk without a manifest save, and in-memory stems
            # diverged from disk). Skip it and keep the batch going.
            try:
                resolved_img_path, dest_img_name = self._save_or_copy_image(
                    clean_stem=clean_stem,
                    norm_split=norm_split,
                    image_path=s.get("image_path"),
                    image_bytes=s.get("image_bytes"),
                    dest_images_dir=dest_images_dir,
                )
            except Exception as ex:
                logger.warning("Skipping bulk sample '%s': image save failed: %s", clean_stem, ex)
                continue

            if resolved_img_path is None and dest_img_name is None:
                continue

            txt_content = "\n".join(lines) + ("\n" if lines else "")
            dest_label_path = dest_labels_dir / f"{clean_stem}.txt"
            try:
                atomic_write_text(dest_label_path, txt_content, encoding="utf-8")
            except Exception as ex:
                logger.warning("Skipping bulk sample '%s': label write failed: %s", clean_stem, ex)
                continue

            # Drop stale files if this stem was previously approved
            # into the other split (new image+label are on disk now).
            self._remove_other_split_files(clean_stem, norm_split)

            # F-0914: approved_at=None (or garbage) must not crash the whole batch.
            _raw_ts = s.get("approved_at")
            try:
                timestamp = float(_raw_ts) if _raw_ts is not None else time.time()
            except (TypeError, ValueError):
                timestamp = time.time()
            pending.append({
                "stem": clean_stem,
                "split": norm_split,
                "image_file": dest_img_name,
                "label_file": f"{clean_stem}.txt",
                "box_count": len(valid_boxes),
                "classes": sorted(list(classes_present)),
                "approved_at": timestamp,
                "source_image": str(resolved_img_path) if resolved_img_path else "",
                "_image_path": dest_images_dir / str(dest_img_name),
                "_label_path": dest_label_path,
            })

        if not pending:
            self.last_bulk_dropped_boxes = dict(dropped_total)
            return 0

        # Phase 2 — single manifest commit under the lock (one manifest
        # write for the whole batch, as before). Entries whose files
        # vanished mid-batch (concurrent remove / external delete) are
        # skipped so the manifest never points at missing files.
        success_count = 0
        with self._lock:
            stems_map = self._manifest.setdefault("stems", {})
            for entry in pending:
                if not entry["_image_path"].is_file() or not entry["_label_path"].is_file():
                    logger.warning(
                        "Skipping bulk sample '%s': files vanished before commit",
                        entry["stem"],
                    )
                    continue
                stems_map[entry["stem"]] = {
                    k: v for k, v in entry.items() if not k.startswith("_")
                }
                self._approved_stems.add(entry["stem"])
                success_count += 1

            if success_count > 0:
                self._recalculate_manifest_stats_and_save()

        self.last_bulk_dropped_boxes = dict(dropped_total)
        return success_count

    # -------------------------------------------------------------------------
    # Fast In-Memory Quarantine & Query API
    # -------------------------------------------------------------------------

    def is_approved(self, stem: str) -> bool:
        """
        Checks if stem is recorded as approved so it NEVER re-enters the pending
        review queue on dialog open/close or disk scan.
        """
        if not stem:
            return False
        clean_stem = Path(stem).stem.strip()
        with self._lock:
            return clean_stem in self._approved_stems

    def get_approved_stems(self) -> Set[str]:
        """Returns a copy of the set of all approved stems."""
        with self._lock:
            return set(self._approved_stems)

    def get_approved_count(self) -> int:
        """Returns the number of approved stems (F-1957 [O3]).

        Replaces the import-time monkey-patch that
        ui/yolo_factory_dialog.py stapled onto this class; D59's
        test_approved_pool_wiring_and_counter_badge calls this directly.
        """
        with self._lock:
            return len(self._approved_stems)

    def get_sample_manifest(self, stem: str) -> Optional[Dict[str, Any]]:
        """Return the manifest entry for a specific stem, or None if not found."""
        clean_stem = Path(stem).stem.strip()
        with self._lock:
            entry = self._manifest.get("stems", {}).get(clean_stem)
            return dict(entry) if entry else None

    def remove_approved_sample(self, stem: str) -> bool:
        """
        Removes an approved sample from the pool (deletes images and labels from disk
        and clears from manifest).
        """
        if not stem:
            return False
        clean_stem = Path(stem).stem.strip()

        with self._lock:
            if clean_stem not in self._approved_stems:
                return False

            stems_map = self._manifest.get("stems", {})
            entry = stems_map.pop(clean_stem, {})
            self._approved_stems.discard(clean_stem)

            # Clean disk files in BOTH splits: a stem re-approved into the
            # other split (or an externally duplicated copy) must not leave
            # an orphan that sync_from_disk resurrects into the pool.
            self._delete_sample_files(
                clean_stem, ("train", "val"), (".jpg", ".png", ".jpeg")
            )

            self._recalculate_manifest_stats_and_save()
            return True

    # -------------------------------------------------------------------------
    # Telemetry & Statistics
    # -------------------------------------------------------------------------

    def get_approved_stats(self) -> Dict[str, Any]:
        """
        Returns telemetry dictionary:
          {"total_approved_images": N, "train_count": M, "val_count": K, "total_boxes": B, ...}
        """
        with self._lock:
            return {
                "total_approved_images": int(self._manifest.get("total_approved_images", 0)),
                "train_count": int(self._manifest.get("train_count", 0)),
                "val_count": int(self._manifest.get("val_count", 0)),
                "total_boxes": int(self._manifest.get("total_boxes", 0)),
                "classes": dict(self._manifest.get("classes", {})),
                "last_updated": float(self._manifest.get("updated_at", 0.0)),
                "pool_root": str(self.pool_root),
                "data_yaml_path": str(self.data_yaml_path),
            }

    def _recalculate_manifest_stats(self) -> bool:
        """Recompute summary counters from the stems map in place.

        Returns True if any stored counter changed (stale manifest), False if
        the stored counters already agreed with the stems. Callers hold
        ``self._lock``; this method itself only touches ``self._manifest``.
        """
        stems_map = self._manifest.get("stems", {})
        total_images = len(stems_map)
        train_count = 0
        val_count = 0
        total_boxes = 0
        class_counts: Dict[str, int] = {}

        for entry in stems_map.values():
            if entry.get("split", "train") == "val":
                val_count += 1
            else:
                train_count += 1
            total_boxes += entry.get("box_count", 0)
            for cid in entry.get("classes", []):
                s_cid = str(cid)
                class_counts[s_cid] = class_counts.get(s_cid, 0) + 1

        old = (
            self._manifest.get("total_approved_images"),
            self._manifest.get("train_count"),
            self._manifest.get("val_count"),
            self._manifest.get("total_boxes"),
            self._manifest.get("classes"),
        )
        self._manifest["version"] = "1.0"
        self._manifest["updated_at"] = time.time()
        self._manifest["total_approved_images"] = total_images
        self._manifest["train_count"] = train_count
        self._manifest["val_count"] = val_count
        self._manifest["total_boxes"] = total_boxes
        self._manifest["classes"] = class_counts
        return old != (
            total_images,
            train_count,
            val_count,
            total_boxes,
            class_counts,
        )

    def _backup_manifest_before_write(self) -> None:
        """P0-6 (K-23 / F057-K2): manifest toptan ezilmeden önce `.bak` yaz.

        ``approved_manifest.json`` sync'te toptan yeniden yazılır; yazım
        yarıda kalırsa / kapı yanlışlıkla bir kereyi düşürürse önceki
        manifest geri getirilemezdi. Yedek best-effort'tur: yazılamazsa
        uyarı loglanır, asıl yazım engellenmez.

        NOTE (P0-2 / K-08): kalıcı yıkıcı-yazım yedeği geldiğinde bu
        yerel yardımcı onun paylaşılan ``backup_file`` fonksiyonunu
        çağırmalı — aynı deseni iki yerde tutma.
        """
        if not self.manifest_path.is_file():
            return
        backup_path = self.manifest_path.with_name(self.manifest_path.name + ".bak")
        try:
            shutil.copy2(self.manifest_path, backup_path)
        except Exception as ex:
            logger.warning(
                "Manifest .bak yazılamadı (%s): %s", backup_path, ex
            )

    def _recalculate_manifest_stats_and_save(self) -> None:
        """Internal helper to calculate summary stats and write manifest atomically."""
        self._recalculate_manifest_stats()
        manifest_json = json.dumps(self._manifest, indent=2, ensure_ascii=False)
        self._backup_manifest_before_write()
        atomic_write_text(self.manifest_path, manifest_json, encoding="utf-8")

    # -------------------------------------------------------------------------
    # Disk Re-Synchronization
    # -------------------------------------------------------------------------

    def _find_source_label_txt(self, stem: str, split: str) -> Optional[Path]:
        """P0-6: find the source-dataset TXT for a pool stem, if any.

        The pool TXT is freeze output (always 5-field); for the
        ``hand_label_txt`` origin path the SOURCE TXT under
        ``dataset/labels/<split>/<stem>.txt`` is the provenance
        evidence for a TXT-only hand label. The pool split is tried
        first, then the other split (a freeze rebalance never moves
        TXT-only frames, but a hand-moved file must not fail the
        gate on split mismatch alone).

        NOTE (P0-6 fixup): the pool TXT itself is NOT consulted
        here — pool membership + a valid pool TXT is its own origin
        (``origin="pool"``) in ``sync_from_disk``; this helper only
        serves the TXT-only hand-label path.
        """
        src_labels = self.project_root / "dataset" / "labels"
        for cand_split in (split, "val" if split == "train" else "train"):
            cand = src_labels / cand_split / f"{stem}.txt"
            if cand.is_file():
                return cand
        return None

    def sync_from_disk(self) -> Dict[str, Any]:
        """
        Reconciles in-memory index and `approved_manifest.json` directly from the
        image and label files existing in `dataset/approved/{images,labels}`.

        P0-6 (K-23) origin gate — FIXUP (2026-11-11): the first
        version of the gate was too strict: it treated a pool file
        alone as NO approval evidence, so a frame written into the
        pool by ``add_approved_sample`` / freeze lost its approval
        the moment ``approved_manifest.json`` was deleted (r46:
        ``good_one``; w22: ``disk_guard_1/2`` — their review JSONs
        carry ``"done": null``, so the gate skipped them as
        "review onaysız"). ``sync_from_disk`` scans the APPROVED
        pool directory: a frame standing there with a readable,
        >=1-valid-box label TXT got there through an approval path
        (freeze / ``add_approved_sample``), so pool membership +
        valid label TXT is itself origin evidence
        (``origin="pool"``). A stem is admitted with provenance:
          (a) its review JSON has ``human_approved is True`` (the
              normal approved flow — ``done`` / bare ``approved``
              do NOT count, F-0828), or
          (b) no review JSON exists for the stem AND its
              source-dataset TXT is a reconciled hand label (>=1
              valid exactly-5-field box — the faz-1 predicate,
              ``_read_hand_label_txt``), or
          (c) it is a pool member with a valid label TXT
              (``origin="pool"`` — the frame is already in the
              approved directory with a good label).
        Template-box worry is freeze time's job (faz-1 already
        closes the freeze admission), not sync time's: sync must
        not drop what freeze / the approve path already admitted.
        Anything without a readable label / with 0 valid boxes is
        still skipped, counted and logged — never silent.
        """
        with self._lock:
            self.ensure_pool_structure()
            stems_map: Dict[str, Any] = {}
            approved_set: Set[str] = set()
            duplicates_removed: List[str] = []
            skipped_image_ext: List[str] = []
            skipped_zero_box: List[str] = []
            skipped_label_read_errors: List[str] = []
            # FIXUP: these two lists no longer fill — every frame
            # reaching the gate is admitted (review / hand-label TXT
            # / pool). Kept so the stats keys
            # skipped_origin_unknown(_count) and
            # skipped_review_not_approved(_count) still exist (0 /
            # []) for callers reading them.
            skipped_origin_unknown: List[str] = []
            skipped_review_not_approved: List[str] = []
            origin_review_count = 0
            origin_hand_label_txt_count = 0
            origin_pool_count = 0

            # P0-6: origin evidence is read ONCE per sync, from the review
            # dir — review approval / hand-label TXT take precedence;
            # pool membership + valid label is the fallback origin.
            review_dir = self.project_root / "dataset" / "review"
            review_approved_stems, review_stems, review_parse_errors = (
                _review_approval_stems(review_dir)
            )
            if not review_dir.is_dir():
                logger.warning(
                    "sync_from_disk: review dizini yok (%s) — review "
                    "onayı aranmaz, havuz üyeliği + geçerli etiket "
                    "köken sayılır.",
                    review_dir,
                )


            for split, img_dir, lbl_dir in [
                ("train", self.images_train_dir, self.labels_train_dir),
                ("val", self.images_val_dir, self.labels_val_dir),
            ]:
                if not img_dir.is_dir():
                    continue

                for img_file in img_dir.iterdir():
                    suf = img_file.suffix.lower()
                    if suf not in (".jpg", ".png", ".jpeg"):
                        # B-11: an image-like file the pool never registers
                        # (S-4601) must not vanish silently — count it and
                        # report it once per sync (see the summary below).
                        if suf in SKIPPED_IMAGE_EXTS:
                            skipped_image_ext.append(f"{split}/{img_file.name}")
                        continue
                    stem = img_file.stem

                    # F-1451/S-1903: suffix alone does not make a file an
                    # image — skip entries whose header is not a real image
                    # instead of registering corrupt bytes as approved data.
                    if not _image_file_magic_ok(img_file):
                        logger.warning(
                            "Skipping non-image file in approved pool: %s",
                            img_file,
                        )
                        continue

                    # B-08: the same stem in both splits is train/val
                    # leakage — the first (train) registration wins and
                    # the later split's copy is removed, like bulk_add's
                    # _remove_other_split_files. A manifest that says
                    # "one split" while the disk keeps both is a lie:
                    # data.yaml serves the directories, not the manifest,
                    # so the stale copy would still train as val.
                    if stem in stems_map and str(stems_map[stem].get("split")) != split:
                        keep_split = str(stems_map[stem].get("split", "train"))
                        self._remove_other_split_files(stem, keep_split=keep_split)
                        duplicates_removed.append(
                            f"{stem} ({keep_split} kept, {split} removed)"
                        )
                        logger.warning(
                            "sync_from_disk: '%s' iki split'te de vardı — %s kopyası silindi, %s korundu.",
                            stem,
                            split,
                            keep_split,
                        )
                        continue

                    lbl_file = lbl_dir / f"{stem}.txt"
                    boxes_count = 0
                    classes_in_file: Set[int] = set()
                    label_read_error = False
                    if lbl_file.is_file():
                        try:
                            lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
                            for line in lines:
                                norm = normalize_box(line)
                                if norm is not None:
                                    boxes_count += 1
                                    classes_in_file.add(norm[0])
                        except Exception as ex:
                            # P0-6 (K-23): okunamayan etiket artık sessiz
                            # geçilmiyor — sayılır + loglanır (21:24'teki
                            # 0-bayt TXT'ler bir sonraki sync'te kayıptan
                            # sayılacak).
                            label_read_error = True
                            skipped_label_read_errors.append(f"{split}/{stem}")
                            logger.warning(
                                "sync_from_disk: etiket okunamadı, kare atlandı %s: %s",
                                lbl_file,
                                ex,
                            )

                    if label_read_error:
                        continue

                    # F-0915: an image without a label (or with zero valid
                    # boxes) is not approved data — do not register it.
                    # P0-6: 0-kutu atlaması sayılır + loglanır.
                    if boxes_count == 0:
                        skipped_zero_box.append(f"{split}/{stem}")
                        logger.debug(
                            "sync_from_disk: 0 geçerli kutu, kare atlandı %s/%s",
                            split,
                            stem,
                        )
                        continue

                    # P0-6 origin gate (K-23) — FIXUP (2026-11-11):
                    # pool membership + valid label TXT is itself
                    # origin evidence (origin="pool"). A frame that
                    # reached this point has a readable label with
                    # >=1 valid box (0-box / unreadable labels were
                    # skipped above) and stands in the APPROVED pool
                    # directory — it got there through an approval
                    # path (freeze / add_approved_sample). Review
                    # approval and TXT hand-label origins still take
                    # precedence, so their stats keep their meaning.
                    if stem in review_approved_stems:
                        origin = "review"
                        origin_review_count += 1
                    elif stem not in review_stems:
                        src_txt = self._find_source_label_txt(stem, split)
                        if src_txt is not None and _is_hand_label_txt(src_txt):
                            origin = "hand_label_txt"
                            origin_hand_label_txt_count += 1
                        else:
                            origin = "pool"
                            origin_pool_count += 1
                    else:
                        # Review JSON var ama insan onayı yok (w22:
                        # "done": null) — TXT yolu bu stem için
                        # kapalıdır (faz-1 ile aynı kural), ama havuz
                        # üyeliği + geçerli etiket başlı başına köken
                        # kanıtıdır: kare onay yoluyla havuza girmiştir.
                        origin = "pool"
                        origin_pool_count += 1
                    approved_set.add(stem)

                    stems_map[stem] = {
                        "stem": stem,
                        "split": split,
                        "image_file": img_file.name,
                        "label_file": f"{stem}.txt",
                        "box_count": boxes_count,
                        "classes": sorted(list(classes_in_file)),
                        "approved_at": img_file.stat().st_mtime,
                        "source_image": str(img_file),
                        "origin": origin,
                    }

            self._manifest["stems"] = stems_map
            self._approved_stems = approved_set
            self._recalculate_manifest_stats_and_save()
            stats = self.get_approved_stats()
            stats["origin_review_count"] = origin_review_count
            stats["origin_hand_label_txt_count"] = origin_hand_label_txt_count
            stats["origin_pool_count"] = origin_pool_count
            stats["skipped_zero_box_count"] = len(skipped_zero_box)
            stats["skipped_zero_box"] = skipped_zero_box[:10]
            stats["skipped_label_read_error_count"] = len(skipped_label_read_errors)
            stats["skipped_label_read_errors"] = skipped_label_read_errors[:10]
            stats["skipped_origin_unknown_count"] = len(skipped_origin_unknown)
            stats["skipped_origin_unknown"] = skipped_origin_unknown[:10]
            stats["skipped_review_not_approved_count"] = len(skipped_review_not_approved)
            stats["skipped_review_not_approved"] = skipped_review_not_approved[:10]
            stats["review_parse_errors"] = review_parse_errors
            stats["review_parse_error_count"] = len(review_parse_errors)
            if skipped_zero_box:
                logger.warning(
                    "sync_from_disk: %d kare 0 geçerli kutu yüzünden atlandı: %s",
                    len(skipped_zero_box),
                    ", ".join(skipped_zero_box[:5]),
                )
            if skipped_label_read_errors:
                logger.warning(
                    "sync_from_disk: %d karenin etiketi okunamadı (atlandı): %s",
                    len(skipped_label_read_errors),
                    ", ".join(skipped_label_read_errors[:5]),
                )
            if skipped_origin_unknown:
                logger.warning(
                    "sync_from_disk: %d kare köken kanıtı yok diye havuza alınmadı "
                    "(review onayı + el-etiket TXT yok): %s",
                    len(skipped_origin_unknown),
                    ", ".join(skipped_origin_unknown[:5]),
                )
            if skipped_review_not_approved:
                logger.warning(
                    "sync_from_disk: %d kare review onaysız diye havuza alınmadı: %s",
                    len(skipped_review_not_approved),
                    ", ".join(skipped_review_not_approved[:5]),
                )
            if duplicates_removed:
                stats["duplicate_stems_removed"] = duplicates_removed
            if skipped_image_ext:
                logger.warning(
                    "sync_from_disk: %d görüntü dosyası havuz formatı dışında atlandı (S-4601, yalnızca .jpg/.jpeg/.png kayda alınır): %s",
                    len(skipped_image_ext),
                    ", ".join(skipped_image_ext[:5]),
                )
                stats["skipped_image_ext_count"] = len(skipped_image_ext)
                stats["skipped_image_ext_files"] = skipped_image_ext[:10]
            return stats

    # -------------------------------------------------------------------------
    # YOLO data.yaml Generation (A07)
    # -------------------------------------------------------------------------

    def generate_approved_data_yaml(
        self,
        target_path: Optional[Union[str, Path]] = None,
        class_names: Optional[Dict[int, str]] = None,
    ) -> str:
        """
        Generates valid Ultralytics YOLO `data.yaml` pointing to
        `dataset/approved/images/train` and `dataset/approved/images/val`
        with canonical 27 classes.

        Args:
            target_path: Destination path for data.yaml. Defaults to `dataset/approved/data_approved.yaml`.
            class_names: Optional class names dictionary. Defaults to CANONICAL_NAMES (27 classes).

        Returns:
            str: Posix string path to the generated data.yaml.
        """
        with self._lock:
            if target_path is not None:
                out_path = Path(target_path)
                if not out_path.is_absolute():
                    out_path = (self.project_root / out_path).resolve()
            else:
                out_path = self.data_yaml_path

            classes = class_names if class_names is not None else CANONICAL_NAMES

            # B-09 DÜZELTME (2026-11-10, canlı kanıt): `path:` MUTLAK yol
            # yazar — eski göreli `.` YANLIŞTI. Ultralytics göreli
            # path'i yaml'ın dizinine göre DEĞİL, eğitim sürecinin
            # çalışma klasörüne (CWD, PC'de F:\a2) göre çözer; `.`
            # yazılınca eğitim F:\a2\images\val'i aradı ve "Hata
            # Oluştu" ile düştü. Taşınabilirlik doğruluk karşısında
            # ikincildir: yaml her dondurmada yeniden üretilir, o
            # yüzden makineye özgü mutlak yol taşınabilirlik sorunu
            # çıkarmaz. `train`/`val` mutlak `path`'e göreli kalır.
            # Windows uyumu için forward-slash (as_posix).
            lines: List[str] = [
                "# Ultralytics YOLOv8 / YOLOv11 Dataset Configuration - Approved Dataset Pool",
                "# Division 1 (A04, A07) — Isolated Human-Approved Ground Truth Training Set",
                f"path: {out_path.parent.as_posix()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(classes)}",
                "names:",
            ]

            for cid in sorted(classes.keys()):
                lines.append(f"  {int(cid)}: {_yaml_scalar(classes[cid])}")

            content = "\n".join(lines) + "\n"
            atomic_write_text(out_path, content, encoding="utf-8")
            logger.info("Generated approved data.yaml at: %s (%d classes)", out_path, len(classes))
            return out_path.as_posix()

    # -------------------------------------------------------------------------
    # High-Level Preparation Pipeline (A07)
    # -------------------------------------------------------------------------

    def prepare_approved_training_dataset(
        self,
        source_dataset_dir: Optional[Union[str, Path]] = None,
        val_ratio: float = 0.15,
        force_rebalance: bool = False,
    ) -> Dict[str, Any]:
        """
        Fulfills A07: Scans source dataset review files for all samples
        with an explicit human approval, migrates/syncs them to
        `dataset/approved/`, automatically balances train/val splits
        (85%/15%), and produces `data_approved.yaml`. K-21 / P1-10:
        admission needs BOTH a frame-level ``human_approved is True``
        AND at least one trusted box with per-box human provenance
        (``hand_label`` / ``human_approved`` — see
        ``box_has_human_provenance``). ``done == "ok"`` /
        ``bulk_approved`` alone never admits a sample: bulk approval
        never inspected the boxes one by one.

        Args:
            source_dataset_dir: Root dataset folder (defaults to `project_root / dataset`).
            val_ratio: Proportion of samples allocated to validation (default 0.15 = 15%).
            force_rebalance: If True, re-assigns train/val according to val_ratio for all items.

        Returns:
            Dict[str, Any]: Telemetry summary of the approved training pool.
        """
        # No outer lock here on purpose: the review scan and the bulk
        # I/O phase run lock-free (see bulk_add_approved_samples);
        # bulk_add / generate_approved_data_yaml / get_approved_stats
        # each take the lock only for their manifest commit.
        src_dir = Path(source_dataset_dir).resolve() if source_dataset_dir else self.project_root / "dataset"
        review_dir = src_dir / "review"

        candidates: List[Dict[str, Any]] = []
        review_parse_errors: List[str] = []
        # P1-09 (K-24): dondurma atamasını kalıcılaştırmak için onaylı
        # adayın review JSON yolu stem ile tutulur; okunamayan review
        # JSON'larının stem'leri budamada fail-closed tutulur (onayı
        # bilinmeyen kare havuzdan atılmaz).
        review_json_by_stem: Dict[str, Path] = {}
        unparseable_review_stems: Set[str] = set()
        # TXT el-etiket taraması için: review JSON'u olan tüm stem'ler
        # (onaylı olsun olmasın) — bu stem'lere TXT yolu hiç dokunmaz.
        review_stems: Set[str] = set()

        # A missing review dir (or an unreadable review file) must not
        # look like a successful empty preparation (sweep E-08): surface
        # it in the returned stats instead of swallowing it in debug.
        review_dir_missing = not review_dir.is_dir()
        if review_dir_missing:
            logger.warning(
                "Onay havuzu hazırlığı: review dizini yok (%s) — 0 aday ile boş sonuç dönecek.",
                review_dir,
            )

        if review_dir.is_dir():
            # S275 (v19/P3-8): rglob sırası dosya sistemine bağlıydı ve
            # aynı stem iki alt dizinde (train/val) varsa son yazan
            # sessizce kazanıyordu — seçim belirsizdi. Sıralı tara,
            # çakışmada İLKini tut ve uyar.
            for json_file in sorted(review_dir.rglob("*.json")):
                if json_file.name in ("trusted_seed.json", "manifest_stats.json"):
                    continue
                # Dosya varlığı bile TXT yolunu kapatır (okunamasa da).
                review_stems.add(json_file.stem)
                try:
                    payload = json.loads(json_file.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        continue

                    # F-0828: only an explicit human approval counts.
                    # `done == "ok"` / `approved is True` are pipeline
                    # flags, not human approval, and must not admit a
                    # sample into the approved pool. K-21 / P1-10: the
                    # frame-level flag alone is not enough either — at
                    # least one trusted box must carry per-box human
                    # provenance (`hand_label` / `human_approved`).
                    # A bulk approval (`bulk_approved`, `done="ok"` with
                    # uninspected boxes) admits nothing on its own.
                    # P3-27: okuma tek kanonik yoldan
                    # (``approval_state.is_frame_human_approved`` +
                    # ``frame_has_provenance_box``) — sync kapısı
                    # (``_review_approval_stems``) AYNI fonksiyonları okur.
                    # NOT (v17 birleşim regresyonu, ba5bec6): P3-27
                    # birleşimi kapıyı kanonik fonksiyonlara çevirirken
                    # ``trusted = payload.get("trusted", [])`` atamasını
                    # sildi ama aşağıdaki aday dict'inde
                    # ``"trusted_boxes": trusted`` kullanımı kaldı — kapıdan
                    # geçen her aday ``NameError`` ile atıldı, havuz 0
                    # kaldı. ``frame_has_provenance_box`` AYNI listeyi
                    # okur (approval_state.py:233), yani bu atama kapıyı
                    # gevşetmez; kapının kabul ettiği kutuları adaya taşır.
                    trusted = payload.get("trusted", [])
                    is_ok = is_frame_human_approved(payload) and frame_has_provenance_box(payload)
                    if not is_ok:
                        continue

                    stem = payload.get("stem") or json_file.stem
                    if isinstance(payload.get("stem"), str) and payload["stem"].strip():
                        review_stems.add(payload["stem"].strip())
                    img_path = payload.get("image_path")
                    split = payload.get("split", "train")

                    candidates.append({
                        "stem": stem,
                        "split": split,
                        "image_path": img_path,
                        "trusted_boxes": trusted,
                        "approved_at": payload.get("approved_at", json_file.stat().st_mtime),
                    })
                    if str(stem) in review_json_by_stem:
                        logger.warning(
                            "Onay havuzu: stem çakışması — '%s' hem %s hem %s'de var; "
                            "ilki (%s) kullanılacak.",
                            stem, review_json_by_stem[str(stem)], json_file,
                            review_json_by_stem[str(stem)],
                        )
                    else:
                        review_json_by_stem[str(stem)] = json_file
                except Exception as ex:
                    review_parse_errors.append(str(json_file))
                    unparseable_review_stems.add(json_file.stem)
                    logger.warning("Review JSON okunamadı, aday sayılmadı %s: %s", json_file, ex)

        # --- TXT el-etiket taraması (ikinci yol) ---------------------------
        # Tarama (ui/yolo_factory_dialog.py::_scan_review_queue_items)
        # review JSON'u yok + labels TXT var ise kutuyu hand_label=True /
        # trusted / source="hand_label_txt" sayar ve Budama/Reddet/toplu
        # onay bunlara dokunamaz — ama dondurma yalnızca review JSON
        # taradığı için bu kareler havuza hiç girmiyordu. Kullanıcı
        # kararı (2026-11-10): dondurma TXT kökenli el etiketli kareleri
        # onaylanmış saysın.
        # Predicate:
        #   - images/{train,val} altında görsel var;
        #   - aynı stem'e ait review JSON YOK (varsa bu yol dokunmaz:
        #     onaysız model kutulu kare havuza girmez, çift sayım yok);
        #   - labels/<split>/<stem>.txt var ve en az 1 GEÇERLİ 5 ALANLI
        #     kutu içeriyor (boş / geçersiz TXT => W46: boş kare onaylanmaz).
        #     DONDURMA ≠ DİALOG KORUMASI: dialog
        #     (ui/yolo_factory_dialog.py::_scan_review_queue_items) 6 alanlı
        #     satırı da el etiketi sayıp Budama/Reddet'e karşı korur, ama
        #     dondurma (onay) daha katıdır — yalnızca TAM 5 alanlı saf YOLO
        #     satırı el etiketidir. 6+ alanlı satır model çıktısıdır (6. alan
        #     güven skorudur; dataset_factory/atomic_review_sync.py:714,
        #     dataset_factory/label_trust.py:1642) ve asla onaylanmaz.
        # split görselin dizininden alınır; force_rebalance bu adaylara
        # uygulanmaz (görselin bulunduğu split korunur).
        txt_candidates: List[Dict[str, Any]] = []
        txt_read_errors: List[str] = []
        txt_model_lines_skipped: int = 0
        seen_txt_stems: Set[str] = set()
        for split in ("train", "val"):
            split_images_dir = src_dir / "images" / split
            if not split_images_dir.is_dir():
                continue
            for img_file in sorted(split_images_dir.iterdir()):
                if not img_file.is_file():
                    continue
                if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                stem = img_file.stem
                if stem in review_stems or stem in seen_txt_stems:
                    continue
                seen_txt_stems.add(stem)
                label_path = src_dir / "labels" / split / f"{stem}.txt"
                if not label_path.is_file():
                    continue
                try:
                    # P0-6: faz-1 predikatı tek fonksiyonda —
                    # _read_hand_label_txt (yalnızca TAM 5 alanlı satırlar
                    # el etiketi kutusudur; 6+ alanlı satırlar model
                    # çıktısıdır, 6. alan güven skorudur).
                    hand_lines, valid_boxes, model_skipped = _read_hand_label_txt(label_path)
                except Exception as ex:
                    txt_read_errors.append(str(label_path))
                    logger.warning("El etiket TXT okunamadı, aday sayılmadı %s: %s", label_path, ex)
                    continue
                txt_model_lines_skipped += model_skipped
                # W46: boş kare onaylanmaz — geçerli 5 alanlı kutusu
                # olmayan TXT atlanır (yalnızca model satırlı kare de
                # burada elenir).
                # (bulk_add_approved_samples da zero-box atlar — F-0784 — bu
                # tarama birinci güvenlik ağıdır.)
                if not valid_boxes:
                    continue
                txt_candidates.append({
                    "stem": stem,
                    "split": split,
                    "image_path": str(img_file),
                    "trusted_boxes": hand_lines,
                    "approved_at": label_path.stat().st_mtime,
                })
        txt_candidates.sort(key=lambda x: str(x["stem"]))

        # Sort deterministically by stem
        candidates.sort(key=lambda x: str(x["stem"]))

        # Rebalance train/val split if requested or needed.
        # P1-09 (K-24 / F063-K2): eski Bresenham ataması stem-sıralı
        # `idx`'e bağlıydı — tek kare ekleme/çıkarma ~2×val_count karenin
        # split'ini değiştiriyordu. Rebalance artık stem-hash tabanlıdır
        # (`_resolve_split`, val_ratio'yu hesaba katar): bir karenin
        # split'i yalnızca kendi stem'ine ve orana bağlıdır, başka kare
        # eklenip çıkarılması onu oynatmaz. Kesin val sayısı garantisi
        # yoktur (hash beklenen oranı verir); oran, kaydırıcıdan gelir.
        if force_rebalance and candidates:
            ratio = min(0.9, max(0.0, float(val_ratio)))
            for c in candidates:
                c["split"] = self._resolve_split(str(c["stem"]), "", val_ratio=ratio)

        # Add candidates to pool. TXT el etiket adayları rebalance
        # dışındadır (split görsel dizininden gelir); review adaylarıyla
        # birlikte toplu eklenir.
        txt_stems = {str(c["stem"]) for c in txt_candidates}
        already_approved_txt = txt_stems & self.get_approved_stems()
        candidates_added = self.bulk_add_approved_samples(
            candidates + txt_candidates, val_ratio=val_ratio
        )
        # v19-pool-drop-count: bu turda havuz TXT'sine yazılmadan elenen
        # kutular — bulk int döndürür, sayaç instance'dan alınır.
        pool_dropped_by_reason = dict(self.last_bulk_dropped_boxes)
        if pool_dropped_by_reason:
            logger.warning(
                "Onay havuzu hazırlığı: %d kutu havuz TXT'sine yazılmadan "
                "elendi: %s",
                sum(pool_dropped_by_reason.values()),
                ", ".join(
                    f"{reason}={count}"
                    for reason, count in sorted(pool_dropped_by_reason.items())
                ),
            )
        txt_added = len((txt_stems & self.get_approved_stems()) - already_approved_txt)
        if txt_candidates:
            logger.info(
                "Onay havuzu hazırlığı: %d TXT el etiket adayı bulundu, %d havuza eklendi.",
                len(txt_candidates), txt_added,
            )
        if txt_model_lines_skipped:
            logger.info(
                "Onay havuzu hazırlığı: %d model satırı (6+ alanlı, güven "
                "skorlu) atlandı — dondurma yalnızca 5 alanlı el etiketi "
                "satırını alır.",
                txt_model_lines_skipped,
            )

        # P1-09 (K-24 / F059-B2): split atamasını review JSON'a yaz.
        # Dondurmada val'a taşınan kare, atama yalnızca bellekteyse bir
        # sonraki prepare'da review JSON'daki eski "train" (ya da eksik
        # split → varsayılan "train") yüzünden train'e geri taşınıyordu.
        # Havuz manifestindeki GERÇEK split review JSON'a kalıcı yazılır;
        # yalnızca değeri değişen dosyalara dokunulur.
        split_persisted = 0
        split_persist_errors: List[str] = []
        for cand in candidates:
            stem = str(cand["stem"])
            json_file = review_json_by_stem.get(stem)
            if json_file is None:
                continue
            entry = self.get_sample_manifest(stem)
            if not entry:
                continue
            effective_split = str(entry.get("split") or "").lower().strip()
            if effective_split not in ("train", "val"):
                continue
            try:
                payload = json.loads(json_file.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    continue
                if str(payload.get("split") or "").lower().strip() == effective_split:
                    continue
                payload["split"] = effective_split
                atomic_write_text(
                    json_file,
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                split_persisted += 1
            except Exception as ex:
                split_persist_errors.append(str(json_file))
                logger.warning(
                    "Split ataması review JSON'a yazılamadı %s: %s", json_file, ex
                )
        if split_persisted:
            logger.info(
                "Onay havuzu hazırlığı: %d karenin split ataması review JSON'a yazıldı.",
                split_persisted,
            )

        # P1-09 (K-24 / F059-B3, F060-K1): prepare yalnızca ekliyordu —
        # onayı kalkan/silinen karenin havuz kopyası kalıyor, eğitiliyordu.
        # Havuzda olup artık onaylı-olmayan stem'ler budanır (dosya +
        # manifest). Korunur (fail-closed): TXT el etiket adayları,
        # okunamayan review JSON'lu stem'ler ve bu turda onaylı bulunanlar.
        keep_stems: Set[str] = (
            {str(c["stem"]) for c in candidates}
            | txt_stems
            | set(unparseable_review_stems)
        )
        pruned_stems: List[str] = []
        for pool_stem in sorted(self.get_approved_stems()):
            if pool_stem in keep_stems:
                continue
            try:
                if self.remove_sample(pool_stem):
                    pruned_stems.append(pool_stem)
            except Exception as ex:
                logger.warning("Havuz budama başarısız %s: %s", pool_stem, ex)
        if pruned_stems:
            logger.info(
                "Onay havuzu hazırlığı: onayı kalkmış/silinen %d kare havuzdan budandı: %s",
                len(pruned_stems),
                ", ".join(pruned_stems[:10]),
            )

        # Generate YOLO configuration
        self.generate_approved_data_yaml()

        stats = self.get_approved_stats()
        # Preparation failures must be visible in the result, not hidden
        # behind a success-looking empty stats dict (sweep E-08).
        stats["candidates_found"] = len(candidates)
        stats["candidates_added"] = int(candidates_added)
        # v19-pool-drop-count: havuz TXT yazımında elenen kutular
        # (geçersiz kutu hâlâ yazılmaz — yalnızca sayılır/loglanır).
        stats["dropped_pool_boxes"] = int(sum(pool_dropped_by_reason.values()))
        stats["dropped_pool_boxes_by_reason"] = dict(pool_dropped_by_reason)
        stats["txt_hand_label_candidates_found"] = len(txt_candidates)
        stats["txt_hand_label_candidates_added"] = int(txt_added)
        stats["txt_model_lines_skipped"] = int(txt_model_lines_skipped)
        stats["txt_hand_label_read_errors"] = txt_read_errors
        stats["txt_hand_label_read_error_count"] = len(txt_read_errors)
        stats["review_parse_errors"] = review_parse_errors
        stats["review_parse_error_count"] = len(review_parse_errors)
        stats["review_dir_missing"] = bool(review_dir_missing)
        stats["split_persisted_count"] = int(split_persisted)
        stats["split_persist_errors"] = split_persist_errors
        stats["split_persist_error_count"] = len(split_persist_errors)
        stats["pruned_count"] = len(pruned_stems)
        stats["pruned_stems"] = list(pruned_stems)
        if review_parse_errors:
            logger.warning(
                "Onay havuzu hazırlığı: %d review JSON okunamadı (atlandı).",
                len(review_parse_errors),
            )
        return stats


# =============================================================================
# Global Singleton & Top-Level Module API
# =============================================================================

_GLOBAL_POOL: Optional[ApprovedDatasetPool] = None
_GLOBAL_LOCK = threading.Lock()


def get_approved_pool(
    project_root: Optional[Union[str, Path]] = None,
    pool_root: Optional[Union[str, Path]] = None,
) -> ApprovedDatasetPool:
    """Returns or initializes the thread-safe global `ApprovedDatasetPool` singleton."""
    global _GLOBAL_POOL
    with _GLOBAL_LOCK:
        if _GLOBAL_POOL is not None and (pool_root is not None or project_root is not None):
            # F-0829: an explicitly requested pool_root that differs from the
            # singleton's root must be honoured (rebind), never silently
            # ignored. A bare project_root mismatch is only warned about:
            # the singleton may have been deliberately pre-seeded with a
            # custom pool_root (dialog wiring), and project_root alone does
            # not determine the pool root — rebinding there would silently
            # move the pool to <project>/dataset/approved instead.
            if pool_root is not None:
                requested_pool = Path(pool_root).resolve()
                if requested_pool != _GLOBAL_POOL.pool_root.resolve():
                    logger.info(
                        "Approved pool singleton rebound: %s -> %s",
                        _GLOBAL_POOL.pool_root,
                        requested_pool,
                    )
                    _GLOBAL_POOL = ApprovedDatasetPool(
                        pool_root=pool_root, project_root=project_root
                    )
            elif project_root is not None:
                _proj = Path(project_root).resolve()
                if _proj.name == "dataset":
                    _proj = _proj.parent
                if _proj != _GLOBAL_POOL.project_root.resolve():
                    logger.warning(
                        "get_approved_pool(project_root=%s) called but the "
                        "singleton is bound to %s (pool_root=%s); keeping "
                        "the existing pool — pass pool_root explicitly to rebind.",
                        _proj,
                        _GLOBAL_POOL.project_root,
                        _GLOBAL_POOL.pool_root,
                    )
        if _GLOBAL_POOL is None:
            _GLOBAL_POOL = ApprovedDatasetPool(pool_root=pool_root, project_root=project_root)
        return _GLOBAL_POOL


def is_approved(stem: str) -> bool:
    """Check if stem is already in the approved dataset pool."""
    return get_approved_pool().is_approved(stem)


def get_approved_stems() -> Set[str]:
    """Return all stems currently in the approved dataset pool."""
    return get_approved_pool().get_approved_stems()


def add_approved_sample(
    stem: str,
    split: str = "train",
    image_path: Optional[Union[str, Path, Any]] = None,
    trusted_boxes: Optional[Sequence[Any]] = None,
    image_bytes: Optional[bytes] = None,
) -> bool:
    """Add a verified sample to the approved dataset pool."""
    return get_approved_pool().add_approved_sample(
        stem=stem,
        split=split,
        image_path=image_path,
        trusted_boxes=trusted_boxes,
        image_bytes=image_bytes,
    )


def remove_approved_sample(stem: str) -> bool:
    """Remove a sample from the approved dataset pool."""
    return get_approved_pool().remove_sample(stem)


def generate_approved_data_yaml(target_path: str = "dataset/approved/data_approved.yaml") -> str:
    """Generate canonical 27-class data_approved.yaml."""
    return get_approved_pool().generate_approved_data_yaml(target_path=target_path)


def get_approved_stats() -> Dict[str, Any]:
    """Return summary telemetry of the approved pool."""
    return get_approved_pool().get_approved_stats()


def prepare_approved_training_dataset(
    source_dataset_dir: Optional[Union[str, Path]] = None,
    val_ratio: float = 0.15,
    force_rebalance: bool = False,
) -> Dict[str, Any]:
    """Execute the preparation pipeline for approved training dataset (A07)."""
    return get_approved_pool().prepare_approved_training_dataset(
        source_dataset_dir=source_dataset_dir,
        val_ratio=val_ratio,
        force_rebalance=force_rebalance,
    )


def check_frozen_manifest_consistency(
    dataset_dir: Optional[Union[str, Path]] = None,
    pool: Optional["ApprovedDatasetPool"] = None,
) -> Dict[str, Any]:
    """Compare `dataset/frozen_manifest.json` against the live approved pool.

    P1-09 (K-24): the training gate used to look only at the `frozen`
    flag, so a freeze followed by new approvals silently trained on a
    different set than the frozen one (the manifest went stale). This
    check compares the manifest's frozen counts with the pool's current
    counts; on mismatch the caller must refuse to start training.

    Returns a dict with:
      - ``checked``: False when no comparison was possible (no manifest,
        unreadable manifest, or a raw-dataset freeze whose counts describe
        the raw dataset, not the pool) — the caller warns but does not block.
      - ``ok``: True when checked and all counts match.
      - ``reason``: human-readable explanation (Turkish).
      - ``mismatches``: list of ``"field: manifest=X pool=Y"`` strings.
      - ``manifest`` / ``pool``: the compared count dicts (or None).
    """
    result: Dict[str, Any] = {
        "checked": False,
        "ok": True,
        "reason": "",
        "mismatches": [],
        "manifest": None,
        "pool": None,
    }
    try:
        if pool is not None:
            ds_dir = Path(getattr(pool, "pool_root")).parent
        elif dataset_dir is not None:
            ds_dir = Path(dataset_dir)
            if ds_dir.name == "approved":
                ds_dir = ds_dir.parent
        else:
            ds_dir = Path("dataset")
        manifest_path = ds_dir / "frozen_manifest.json"
        if not manifest_path.is_file():
            result["reason"] = (
                "frozen_manifest.json yok — dondurma karşılaştırması yapılamadı."
            )
            return result
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            result["reason"] = f"frozen_manifest.json okunamadı ({exc}) — karşılaştırma yapılamadı."
            return result
        if not isinstance(payload, dict):
            result["reason"] = "frozen_manifest.json geçersiz (dict değil) — karşılaştırma yapılamadı."
            return result
        # Raw-mode freezes describe the raw dataset, not the pool —
        # pool counts can never match them; only approved_pool freezes
        # are comparable.
        if str(payload.get("source", "")) != "approved_pool":
            result["reason"] = (
                "Dondurma ham veri seti modunda yapılmış (source != approved_pool) "
                "— havuz karşılaştırması uygulanmaz."
            )
            return result
        if pool is None:
            # Scoped pool on purpose: the singleton may be bound elsewhere
            # or hold stale in-memory state — the check compares the frozen
            # manifest against the on-disk pool manifest.
            pool = ApprovedDatasetPool(
                pool_root=ds_dir / "approved", project_root=ds_dir.parent
            )
        pool_stats = pool.get_approved_stats()
        manifest_counts = {
            "total_frames": int(payload.get("total_frames", 0) or 0),
            "train_frames": int(payload.get("train_frames", 0) or 0),
            "val_frames": int(payload.get("val_frames", 0) or 0),
            "total_boxes": int(payload.get("total_boxes", 0) or 0),
        }
        pool_counts = {
            "total_frames": int(pool_stats.get("total_approved_images", 0) or 0),
            "train_frames": int(pool_stats.get("train_count", 0) or 0),
            "val_frames": int(pool_stats.get("val_count", 0) or 0),
            "total_boxes": int(pool_stats.get("total_boxes", 0) or 0),
        }
        result["manifest"] = dict(manifest_counts)
        result["pool"] = dict(pool_counts)
        result["checked"] = True
        mismatches = [
            f"{field}: dondurulan={manifest_counts[field]} havuz={pool_counts[field]}"
            for field in ("total_frames", "train_frames", "val_frames", "total_boxes")
            if manifest_counts[field] != pool_counts[field]
        ]
        result["mismatches"] = mismatches
        if mismatches:
            result["ok"] = False
            result["reason"] = (
                "Dondurulan veri seti havuzla uyuşmuyor (dondurma sonrası onay/budama "
                "olmuş olabilir) — eğitim başlatılmadı, önce Adım 3'te yeniden dondurun: "
                + "; ".join(mismatches)
            )
        else:
            result["reason"] = "Dondurulan sayılar havuzla uyuşuyor."
        return result
    except Exception as exc:
        # The check itself must never crash the caller: report as
        # "not checked" so the caller warns instead of blocking on an
        # internal error. A real mismatch still blocks via the normal path.
        logger.warning("frozen_manifest karşılaştırması çalıştırılamadı: %s", exc)
        result["checked"] = False
        result["ok"] = True
        result["reason"] = f"Dondurma karşılaştırması çalıştırılamadı ({exc})."
        return result
