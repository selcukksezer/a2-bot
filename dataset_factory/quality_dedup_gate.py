from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

_MASK64 = (1 << 64) - 1


def _popcount_u64(values: np.ndarray) -> np.ndarray:
    """Vectorized population count of a uint64 array (bit counts per element).

    Byte order does not matter for a popcount: unpacking the little-endian
    bytes and summing the bits gives the same total as any other order.
    """
    return np.unpackbits(
        np.ascontiguousarray(values).view(np.uint8).reshape(-1, 8), axis=1
    ).sum(axis=1)


@dataclass
class QualityGateDecision:
    accepted: bool
    reason: str
    sharpness: float
    dhash: int
    min_hamming_seen: int
    split: str
    validated_annotations: List[Dict[str, Any]] = field(default_factory=list)


class DatasetQualityGate:
    """
    Perceptual 64-bit dHash Deduplicator, Laplacian Motion-Blur Gate,
    YOLO Annotation Validator with Class-Aware NMS, and Deterministic Train/Val Splitter.
    """

    def __init__(
        self,
        min_hamming_distance: int = 5,
        min_sharpness_threshold: float = 35.0,
        val_ratio: float = 0.15,
        max_hash_history: int = 512,
        min_box_area: float = 0.00002,
        nms_iou_threshold: float = 0.65,
    ) -> None:
        self.min_hamming_distance = int(min_hamming_distance)
        self.min_sharpness_threshold = float(min_sharpness_threshold)
        self.val_ratio = float(np.clip(val_ratio, 0.01, 0.50))
        self.max_hash_history = max(16, int(max_hash_history))
        self.min_box_area = float(min_box_area)
        self.nms_iou_threshold = float(nms_iou_threshold)

        self._hash_history: Deque[int] = deque(maxlen=self.max_hash_history)
        self._sample_index: int = 0
        self._class_sample_counts: Dict[int, int] = {}

        self.stats: Dict[str, int] = {
            "total_evaluated": 0,
            "accepted": 0,
            "rejected_blur": 0,
            "rejected_duplicate": 0,
            "rejected_empty_frame": 0,
            "train_count": 0,
            "val_count": 0,
        }

    @staticmethod
    def _to_grayscale(frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None.")
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3:
            channels = frame.shape[2]
            if channels == 1:
                return frame[:, :, 0]
            if channels == 4:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    @staticmethod
    def _compute_dhash_gray(gray: np.ndarray) -> int:
        """dHash over an already-grayscale frame (shared by compute_dhash)."""
        resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        flat_bits = diff.flatten()
        dhash_val = 0
        for bit in flat_bits:
            dhash_val = (dhash_val << 1) | int(bool(bit))
        return int(dhash_val)

    @staticmethod
    def compute_dhash(frame: np.ndarray) -> int:
        """
        Resizes grayscale frame to 9x8 (width=9, height=8) and computes
        64-bit horizontal gradient bits as a Python int.
        """
        gray = DatasetQualityGate._to_grayscale(frame)
        return DatasetQualityGate._compute_dhash_gray(gray)

    @staticmethod
    def hamming_distance(h1: int, h2: int) -> int:
        """
        Computes bit-level Hamming distance between two 64-bit integer hashes.
        """
        xor_val = (int(h1) ^ int(h2)) & ((1 << 64) - 1)
        return int(xor_val.bit_count())

    @staticmethod
    def _measure_sharpness_gray(gray: np.ndarray) -> float:
        """Laplacian variance over an already-grayscale frame."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    @staticmethod
    def measure_sharpness(frame: np.ndarray) -> float:
        """
        Computes Laplacian variance on the grayscale frame to measure sharpness.
        """
        gray = DatasetQualityGate._to_grayscale(frame)
        return DatasetQualityGate._measure_sharpness_gray(gray)

    @staticmethod
    def _iou_from_corners(
        ax1: float, ay1: float, ax2: float, ay2: float,
        bx1: float, by1: float, bx2: float, by2: float,
    ) -> float:
        """IoU between two boxes given as precomputed corner coordinates.

        Bit-identical math to compute_iou_xywh; kept separate so the NMS
        inner loop does not recompute corners for every pair.
        """
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0.0:
            return 0.0

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union_area = area_a + area_b - inter_area
        if union_area <= 1e-12:
            return 0.0
        return float(inter_area / union_area)

    @staticmethod
    def compute_iou_xywh(
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float],
    ) -> float:
        """
        Computes Intersection-over-Union (IoU) between two normalized center-based boxes
        (cx, cy, w, h).
        """
        ax1 = box_a[0] - box_a[2] * 0.5
        ay1 = box_a[1] - box_a[3] * 0.5
        ax2 = box_a[0] + box_a[2] * 0.5
        ay2 = box_a[1] + box_a[3] * 0.5

        bx1 = box_b[0] - box_b[2] * 0.5
        by1 = box_b[1] - box_b[3] * 0.5
        bx2 = box_b[0] + box_b[2] * 0.5
        by2 = box_b[1] + box_b[3] * 0.5

        return DatasetQualityGate._iou_from_corners(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)

    @staticmethod
    def _iou_xywh_against_many(
        box: Tuple[float, float, float, float],
        others: np.ndarray,
    ) -> np.ndarray:
        """Vectorized IoU of one (cx, cy, w, h) box against an (N, 4) array.

        Same float64 operations, in the same order, as compute_iou_xywh, so
        each element is bit-identical to the scalar result (the early
        ``0.0`` returns of the scalar version are reproduced with
        ``np.where``). Empty input returns an empty (0,) array.
        """
        if others.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)
        ax1 = box[0] - box[2] * 0.5
        ay1 = box[1] - box[3] * 0.5
        ax2 = box[0] + box[2] * 0.5
        ay2 = box[1] + box[3] * 0.5

        bx1 = others[:, 0] - others[:, 2] * 0.5
        by1 = others[:, 1] - others[:, 3] * 0.5
        bx2 = others[:, 0] + others[:, 2] * 0.5
        by2 = others[:, 1] + others[:, 3] * 0.5

        inter_w = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
        inter_h = np.maximum(0.0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
        union_area = area_a + area_b - inter_area
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = inter_area / union_area
        return np.where(
            (inter_area <= 0.0) | (union_area <= 1e-12), 0.0, iou
        ).astype(np.float64, copy=False)

    def validate_yolo_annotations(
        self, annotations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Clamps and validates cx, cy, w, h in [0.001, 0.999], removes degenerate tiny boxes
        (w * h < 0.00002), and applies class-aware Non-Maximum Suppression (IoU > 0.65).
        Supports both normalized cx/cy/w/h dicts and pixel-coordinate bbox dicts if image_width/height provided.
        """
        if not annotations:
            return []

        sanitized: List[Dict[str, Any]] = []
        for raw in annotations:
            if not isinstance(raw, dict):
                continue

            try:
                class_id = int(raw.get("class_id", raw.get("cls", 0)))
            except (TypeError, ValueError):
                # D37/F-0811: one broken class_id (e.g. "abc") must not
                # crash the whole quality gate -- skip that annotation.
                continue
            if class_id < 0 or class_id > 26:
                continue

            try:
                cx_raw = raw.get("cx", raw.get("x_center"))
                cy_raw = raw.get("cy", raw.get("y_center"))
                w_raw = raw.get("w", raw.get("width"))
                h_raw = raw.get("h", raw.get("height"))

                if cx_raw is None or cy_raw is None or w_raw is None or h_raw is None:
                    bbox = raw.get("bbox")
                    img_w = float(raw.get("image_width", 1.0))
                    img_h = float(raw.get("image_height", 1.0))
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and img_w > 0 and img_h > 0:
                        x1, y1, x2, y2 = [float(v) for v in bbox]
                        cx_raw = ((x1 + x2) * 0.5) / img_w
                        cy_raw = ((y1 + y2) * 0.5) / img_h
                        w_raw = abs(x2 - x1) / img_w
                        h_raw = abs(y2 - y1) / img_h
                    else:
                        continue

                cx = float(cx_raw)
                cy = float(cy_raw)
                w = float(w_raw)
                h = float(h_raw)
                conf = float(raw.get("confidence", raw.get("conf", raw.get("score", 1.0))))
            except (TypeError, ValueError):
                # D37/F-0811: broken coordinates/confidence skip the box,
                # they never abort the whole validation.
                continue

            if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(w) and np.isfinite(h)):
                continue

            if w <= 0.0 or h <= 0.0:
                continue

            # D37/F-0987: a box outside [0, 1] is dropped, not silently
            # clamped/shifted into "valid" (a 0.001 tolerance absorbs
            # float noise). Only in-tolerance values are clamped below.
            if (
                cx < -0.001 or cx > 1.001
                or cy < -0.001 or cy > 1.001
                or w < 0.0 or w > 1.001
                or h < 0.0 or h > 1.001
            ):
                continue

            # Clamp cx, cy, w, h in [0.001, 0.999]
            clamped_cx = float(np.clip(cx, 0.001, 0.999))
            clamped_cy = float(np.clip(cy, 0.001, 0.999))
            clamped_w = float(np.clip(w, 0.001, 0.999))
            clamped_h = float(np.clip(h, 0.001, 0.999))

            area = clamped_w * clamped_h
            if area < self.min_box_area:
                continue

            entry = dict(raw)
            entry["class_id"] = class_id
            entry["cx"] = round(clamped_cx, 6)
            entry["cy"] = round(clamped_cy, 6)
            entry["w"] = round(clamped_w, 6)
            entry["h"] = round(clamped_h, 6)
            entry["confidence"] = conf
            entry["area"] = area
            sanitized.append(entry)

        if not sanitized:
            return []

        # Group by class_id and apply Class-Aware Non-Maximum Suppression (IoU > 0.65)
        by_class: Dict[int, List[Dict[str, Any]]] = {}
        for item in sanitized:
            cid = int(item["class_id"])
            by_class.setdefault(cid, []).append(item)

        kept_all: List[Dict[str, Any]] = []
        for cid in sorted(by_class.keys()):
            candidates = sorted(
                by_class[cid],
                key=lambda x: (float(x.get("confidence", 1.0)), float(x.get("area", 0.0))),
                reverse=True,
            )
            # Vectorized class-aware NMS: repeatedly take the
            # highest-priority remaining box, keep it, and drop every
            # remaining box whose IoU with it exceeds the threshold.
            # Exactly the reference loop's decisions: candidates are
            # processed in priority order, a box is kept iff no kept
            # box overlaps it above the threshold, and the kept set is
            # built in that same order — but with one vector IoU call
            # per kept box instead of one scalar call per pair.
            # Corner coordinates are likewise computed once (into the
            # boxes array below), never rebuilt per pair.
            boxes = np.array(
                [
                    (
                        float(c["cx"]),
                        float(c["cy"]),
                        float(c["w"]),
                        float(c["h"]),
                    )
                    for c in candidates
                ],
                dtype=np.float64,
            )
            remaining = list(range(len(candidates)))
            kept_class: List[Dict[str, Any]] = []
            while remaining:
                top = remaining[0]
                kept_class.append(candidates[top])
                if len(remaining) == 1:
                    break
                rest_idx = remaining[1:]
                ious = self._iou_xywh_against_many(
                    (
                        float(boxes[top, 0]),
                        float(boxes[top, 1]),
                        float(boxes[top, 2]),
                        float(boxes[top, 3]),
                    ),
                    boxes[rest_idx],
                )
                remaining = [
                    idx
                    for idx, iou in zip(rest_idx, ious)
                    if iou <= self.nms_iou_threshold
                ]
            kept_all.extend(kept_class)

        return kept_all

    @staticmethod
    def choose_split(sample_index: int, val_ratio: float = 0.15) -> str:
        """
        Deterministic & Stratified Train/Val Splitter.
        Routes 85% of samples to 'train' and 15% to 'val' deterministically
        using a low-discrepancy Bresenham accumulator schedule so every window of
        20 samples has exactly 3 'val' and 17 'train' samples.
        """
        idx = max(0, int(sample_index))
        ratio = float(np.clip(val_ratio, 0.0, 1.0))
        if ratio <= 0.0:
            return "train"
        if ratio >= 1.0:
            return "val"

        prev_count = int(np.floor(idx * ratio + 1e-9))
        curr_count = int(np.floor((idx + 1) * ratio + 1e-9))
        if curr_count > prev_count:
            return "val"
        return "train"

    def choose_stratified_split(
        self, annotations: Optional[List[Dict[str, Any]]] = None, val_ratio: Optional[float] = None
    ) -> str:
        """
        Chooses split stratified by dominant class ID if annotations are present,
        falling back to global sample_index.
        """
        ratio = self.val_ratio if val_ratio is None else float(val_ratio)
        if annotations:
            # D37/F-0887: the dominant class is the MOST FREQUENT class,
            # not the smallest class_id (the old `min(...)` made class 0
            # dominant whenever a single class-0 box was present).
            counts: Counter = Counter()
            for a in annotations:
                if not isinstance(a, dict):
                    continue
                try:
                    counts[int(a.get("class_id", 0))] += 1
                except (TypeError, ValueError):
                    continue
            if counts:
                dominant_cls = min(counts, key=lambda c: (-counts[c], c))
                cls_idx = self._class_sample_counts.get(dominant_cls, 0)
                self._class_sample_counts[dominant_cls] = cls_idx + 1
                return self.choose_split(cls_idx, val_ratio=ratio)
        return self.choose_split(self._sample_index, val_ratio=ratio)

    def is_duplicate_hash(self, dhash_val: int) -> Tuple[bool, int]:
        """
        Compares dhash_val against rolling history of saved hashes.
        Returns (is_duplicate, min_hamming_distance_found).

        Vectorized (D52): one XOR + popcount over the whole history
        instead of a Python-level hamming_distance() call per entry.
        Bit-identical result: the probe and every history entry are
        masked to 64 bits exactly like hamming_distance() does.
        """
        if not self._hash_history:
            return False, 64
        probe = int(dhash_val) & _MASK64
        hist = np.fromiter(
            (int(h) & _MASK64 for h in self._hash_history),
            dtype=np.uint64,
            count=len(self._hash_history),
        )
        min_dist = int(_popcount_u64(hist ^ np.uint64(probe)).min())
        return (min_dist < self.min_hamming_distance), min_dist

    def evaluate_frame(
        self,
        frame: np.ndarray,
        annotations: Optional[List[Dict[str, Any]]] = None,
        record_on_accept: bool = True,
    ) -> QualityGateDecision:
        """
        Runs the complete quality & deduplication gate on a candidate frame and its annotations:
        1. Validates frame is non-empty
        2. Measures Laplacian sharpness against min_sharpness_threshold
        3. Computes 64-bit dHash and checks against rolling hash history
        4. Validates and deduplicates YOLO annotations via class-aware NMS
        5. Assigns deterministic train/val split
        """
        self.stats["total_evaluated"] += 1

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            self.stats["rejected_empty_frame"] += 1
            return QualityGateDecision(
                accepted=False,
                reason="empty_frame",
                sharpness=0.0,
                dhash=0,
                min_hamming_seen=0,
                split="train",
                validated_annotations=[],
            )

        # Grayscale is computed once and shared by the sharpness and dHash
        # steps (both previously converted the full frame separately).
        gray = self._to_grayscale(frame)
        sharpness = self._measure_sharpness_gray(gray)
        if sharpness < self.min_sharpness_threshold:
            self.stats["rejected_blur"] += 1
            return QualityGateDecision(
                accepted=False,
                reason=f"blurry_frame(sharpness={sharpness:.2f}<{self.min_sharpness_threshold:.2f})",
                sharpness=sharpness,
                dhash=0,
                min_hamming_seen=64,
                split="train",
                validated_annotations=[],
            )

        dhash_val = self._compute_dhash_gray(gray)
        is_dup, min_dist = self.is_duplicate_hash(dhash_val)
        if is_dup:
            self.stats["rejected_duplicate"] += 1
            return QualityGateDecision(
                accepted=False,
                reason=f"duplicate_frame(hamming={min_dist}<{self.min_hamming_distance})",
                sharpness=sharpness,
                dhash=dhash_val,
                min_hamming_seen=min_dist,
                split="train",
                validated_annotations=[],
            )

        valid_anns = self.validate_yolo_annotations(annotations or [])

        # D37/F-0888: a sharp frame with no (valid) annotations is not
        # "ok" -- it must not enter the dataset as an accepted sample.
        if not annotations or not valid_anns:
            reason = "no_annotations" if not annotations else "no_valid_annotations"
            return QualityGateDecision(
                accepted=False,
                reason=reason,
                sharpness=sharpness,
                dhash=dhash_val,
                min_hamming_seen=min_dist,
                split="train",
                validated_annotations=[],
            )

        # D37/F-0988: the hash is always recorded on acceptance. With
        # record_on_accept=False the same frame used to pass the dedup
        # check again on every call; the flag now only controls whether
        # the acceptance is counted in stats / advances the split index.
        self._hash_history.append(dhash_val)
        if record_on_accept:
            split = self.choose_stratified_split(valid_anns, val_ratio=self.val_ratio)
            self._sample_index += 1
            self.stats["accepted"] += 1
            if split == "val":
                self.stats["val_count"] += 1
            else:
                self.stats["train_count"] += 1
        else:
            split = self.choose_split(self._sample_index, val_ratio=self.val_ratio)

        return QualityGateDecision(
            accepted=True,
            reason="ok",
            sharpness=sharpness,
            dhash=dhash_val,
            min_hamming_seen=min_dist,
            split=split,
            validated_annotations=valid_anns,
        )

    @staticmethod
    def format_yolo_lines(annotations: List[Dict[str, Any]]) -> List[str]:
        """
        Formats validated annotations into standard YOLO .txt lines:
        '<class_id> <cx:.6f> <cy:.6f> <w:.6f> <h:.6f>'
        """
        lines: List[str] = []
        for ann in annotations:
            cid = int(ann["class_id"])
            cx = float(ann["cx"])
            cy = float(ann["cy"])
            w = float(ann["w"])
            h = float(ann["h"])
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        return lines

    def compute_sobel_edge_energy(self, frame: np.ndarray) -> float:
        """Compute mean Sobel gradient magnitude energy of frame via SobelClahePatchNormalizer."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return 0.0
        try:
            from core.patch_normalizer import SobelClahePatchNormalizer
            edge_map = SobelClahePatchNormalizer().compute_sobel_edge_map(frame)
            if edge_map is not None and edge_map.size > 0:
                return round(float(np.mean(edge_map)), 3)
        except Exception:
            pass
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            return round(float(np.mean(cv2.magnitude(gx, gy))), 3)
        except Exception:
            return 0.0

    def reset_history(self) -> None:
        """Clears the rolling hash history."""
        self._hash_history.clear()
