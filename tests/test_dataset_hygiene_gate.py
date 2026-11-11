# -*- coding: utf-8 -*-
"""
Tests for Pre-Training Quality & Hygiene Gate (`dataset_factory/dataset_hygiene_gate.py`).
"""

import base64
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest

from dataset_factory.class_taxonomy import ALL_CANONICAL_CLASSES
from dataset_factory.dataset_hygiene_gate import (
    CANONICAL_CLASS_IDS,
    CANONICAL_CLASS_NAMES,
    audit_dataset_hygiene,
    is_canonical_class,
    is_training_box_verified,
    pre_training_dataset_hygiene_gate,
    validate_and_clamp_box,
)

# Real 8x8 black JPEG (cv2-encoded, decodes to an 8x8 image). Fixtures must
# use genuine image bytes: the previous fixture wrote a bare JPEG SOI header
# followed by zero bytes, which no decoder accepts.
_VALID_JPEG_BYTES = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAIAAgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD+f+iiigD/2Q=="
)


class TestDatasetHygieneGate(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_hygiene_gate_")
        self.root = Path(self.temp_dir)
        self.labels_train = self.root / "labels" / "train"
        self.labels_val = self.root / "labels" / "val"
        self.images_train = self.root / "images" / "train"
        self.images_val = self.root / "images" / "val"
        self.review_train = self.root / "review" / "train"
        self.review_val = self.root / "review" / "val"

        for p in (
            self.labels_train,
            self.labels_val,
            self.images_train,
            self.images_val,
            self.review_train,
            self.review_val,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_image(self, split: str, stem: str) -> Path:
        img_p = self.root / "images" / split / f"{stem}.jpg"
        img_p.write_bytes(_VALID_JPEG_BYTES)
        return img_p

    def test_fixture_images_are_valid_jpegs(self):
        """Guard: fixture images must decode as real JPEGs, not header stubs."""
        img_p = self._create_image("train", "fixture_check")
        raw = img_p.read_bytes()
        self.assertTrue(raw.startswith(b"\xff\xd8\xff"), "missing JPEG SOI marker")
        self.assertTrue(raw.endswith(b"\xff\xd9"), "missing JPEG EOI marker")
        if importlib.util.find_spec("cv2") is not None:
            import cv2
            import numpy as np

            decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(decoded, "fixture JPEG does not decode")
            self.assertEqual(decoded.shape, (8, 8, 3))

    def _create_label(self, split: str, stem: str, lines: list[str]) -> Path:
        lbl_p = self.root / "labels" / split / f"{stem}.txt"
        lbl_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return lbl_p

    def _create_review(self, split: str, stem: str, trusted: list = None, weak: list = None) -> Path:
        rev_p = self.root / "review" / split / f"{stem}.json"
        data = {
            "stem": stem,
            "trusted": trusted or [],
            "weak": weak or [],
        }
        rev_p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return rev_p

    def test_canonical_classes(self):
        """Training ids are the exported canonical taxonomy, 0..28.

        Pin the gate to ALL_CANONICAL_CLASSES so 22..28 cannot be purged
        as invalid while a second, narrower list drifts in.
        """
        self.assertEqual(CANONICAL_CLASS_IDS, ALL_CANONICAL_CLASSES)
        self.assertEqual(len(CANONICAL_CLASS_IDS), 29)
        for cid in range(29):
            self.assertTrue(is_canonical_class(cid), cid)
            self.assertIn(cid, CANONICAL_CLASS_NAMES)
        self.assertFalse(is_canonical_class(-1))
        self.assertFalse(is_canonical_class(29))
        self.assertFalse(is_canonical_class(98))
        self.assertFalse(is_canonical_class(True))
        self.assertFalse(is_canonical_class("invalid"))

    def test_geometry_validation_and_clamping(self):
        """Verify zero-area, unnormalized, and clamped boxes."""
        # Zero area
        valid, coords, reason = validate_and_clamp_box(0.5, 0.5, 0.0, 0.05)
        self.assertFalse(valid)
        self.assertIn("Zero or sub-minimal", reason)

        # Negative area
        valid, coords, reason = validate_and_clamp_box(0.5, 0.5, -0.1, 0.1)
        self.assertFalse(valid)

        # Spurious full screen
        valid, coords, reason = validate_and_clamp_box(0.5, 0.5, 0.99, 0.99)
        self.assertFalse(valid)

        # Wildly out of bounds
        valid, coords, reason = validate_and_clamp_box(1.5, 0.5, 0.1, 0.1)
        self.assertFalse(valid)

        # Minor rounding clamped
        valid, coords, reason = validate_and_clamp_box(1.0001, -0.0001, 0.1, 0.1)
        self.assertTrue(valid)
        self.assertIsNotNone(coords)
        cx, cy, w, h = coords
        self.assertAlmostEqual(cx, 0.9999)
        self.assertAlmostEqual(cy, 0.0001)

    def test_check_1_purges_invalid_class_ids(self):
        """Check 1: Non-canonical class IDs (e.g. 98, 70) are purged from .txt and .json."""
        stem = "test_invalid_cid"
        self._create_image("train", stem)
        self._create_label(
            "train",
            stem,
            [
                "0 0.500000 0.500000 0.100000 0.100000",
                "98 0.400000 0.400000 0.200000 0.200000",  # Invalid!
                "3 0.600000 0.600000 0.150000 0.150000",
            ],
        )
        self._create_review(
            "train",
            stem,
            trusted=[
                {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "confidence": 0.95, "source": "hand_label"},
                {"class_id": 98, "cx": 0.4, "cy": 0.4, "w": 0.2, "h": 0.2, "confidence": 0.90, "source": "exemplar_match"},
                {"class_id": 3, "cx": 0.6, "cy": 0.6, "w": 0.15, "h": 0.15, "confidence": 0.92, "source": "hand_label"},
            ],
        )

        report = pre_training_dataset_hygiene_gate(self.root, auto_clean=True)
        self.assertTrue(report["clean"])
        # Exactly 1 invalid box, counted once in the label file and once in
        # the review JSON — exact counts, so over/under-purging cannot hide.
        self.assertEqual(report["purged_boxes"], 2)
        self.assertEqual(report["invalid_class_ids_purged"], 2)

        # Inspect resulting .txt
        lbl_content = (self.labels_train / f"{stem}.txt").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lbl_content), 2)
        cids = [int(line.split()[0]) for line in lbl_content]
        self.assertIn(0, cids)
        self.assertIn(3, cids)
        self.assertNotIn(98, cids)

        # Inspect resulting review JSON
        rev_data = json.loads((self.review_train / f"{stem}.json").read_text(encoding="utf-8"))
        rev_cids = [b["class_id"] for b in rev_data["trusted"]]
        self.assertNotIn(98, rev_cids)

    def test_check_2_purges_unapproved_boxes_in_train(self):
        """Check 2: Unapproved heuristic boxes in training set are purged and demoted."""
        stem = "boot_aria_sample"
        self._create_image("train", stem)
        self._create_label(
            "train",
            stem,
            [
                "0 0.500000 0.500000 0.100000 0.100000",
                "15 0.450000 0.680000 0.340000 0.320000",  # Unverified heuristic path_ground!
            ],
        )
        self._create_review(
            "train",
            stem,
            trusted=[
                {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "confidence": 0.95, "source": "vision_3d_crest", "trusted": True},
                {
                    "class_id": 15,
                    "cx": 0.45,
                    "cy": 0.68,
                    "w": 0.34,
                    "h": 0.32,
                    "confidence": 0.92,
                    "source": "visual_ground_path_segmenter",
                    "trusted": True,
                },
            ],
        )

        report = pre_training_dataset_hygiene_gate(self.root, auto_clean=True)
        self.assertTrue(report["clean"])
        # F-0931: the 1 unapproved box is purged from the label file (1) and
        # DEMOTED to weak in the review JSON (1) — a demoted box still exists
        # in the dataset, so it must not inflate purged_boxes. The stale
        # assertion purged==2 counted the demotion as a purge.
        self.assertEqual(report["purged_boxes"], 1)
        self.assertEqual(report["demoted_boxes"], 1)
        self.assertEqual(report["unapproved_boxes_purged"], 2)

        # Label .txt should now ONLY have aria (0)
        lbl_content = (self.labels_train / f"{stem}.txt").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lbl_content), 1)
        self.assertTrue(lbl_content[0].startswith("0 "))

        # Review JSON should demote path_ground to weak
        rev_data = json.loads((self.review_train / f"{stem}.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rev_data["trusted"]), 1)
        self.assertEqual(rev_data["trusted"][0]["class_id"], 0)
        self.assertEqual(len(rev_data["weak"]), 1)
        self.assertEqual(rev_data["weak"][0]["class_id"], 15)

    def test_check_3_purges_zero_area_and_corrupt_boxes(self):
        """Check 3: Corrupted lines and zero-area boxes are eliminated."""
        stem = "test_corrupted_boxes"
        self._create_image("train", stem)
        self._create_label(
            "train",
            stem,
            [
                "0 0.500000 0.500000 0.100000 0.100000",
                "3 0.200000 0.200000 0.000000 0.000000",  # Zero area!
                "corrupted line with random words",         # Syntax error!
                "4 0.300000 0.300000 0.050000",             # Missing dimension!
                "5 0.700000 0.700000 0.080000 0.080000",
            ],
        )
        self._create_review(
            "train",
            stem,
            trusted=[
                {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "confidence": 0.95, "source": "hand_label"},
                {"class_id": 5, "cx": 0.7, "cy": 0.7, "w": 0.08, "h": 0.08, "confidence": 0.92, "source": "hand_label"},
            ],
        )

        report = pre_training_dataset_hygiene_gate(self.root, auto_clean=True)
        self.assertTrue(report["clean"])
        # Exactly 3 corrupt lines: zero-area, syntax error, missing dimension.
        self.assertEqual(report["purged_boxes"], 3)
        self.assertEqual(report["corrupted_boxes_purged"], 3)

        lbl_content = (self.labels_train / f"{stem}.txt").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lbl_content), 2)
        cids = [int(line.split()[0]) for line in lbl_content]
        self.assertEqual(cids, [0, 5])

    def test_check_4_mismatched_images_and_orphaned_files(self):
        """Check 4: Orphaned labels without images and missing labels for images are cleaned."""
        # 1. Orphaned label (no image)
        self._create_label("train", "orphan_label", ["0 0.5 0.5 0.1 0.1"])
        self._create_review("train", "orphan_label", trusted=[{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}])

        # 2. Image without label file
        self._create_image("train", "orphan_image")

        report = pre_training_dataset_hygiene_gate(self.root, auto_clean=True)
        self.assertTrue(report["clean"])
        # Exactly 3 mismatches: orphaned label + orphaned review + missing
        # label file for the orphaned image.
        self.assertEqual(report["mismatched_files_fixed"], 3)

        # Orphaned label and review should be deleted
        self.assertFalse((self.labels_train / "orphan_label.txt").exists())
        self.assertFalse((self.review_train / "orphan_label.json").exists())

        # Missing label for orphan_image should be created as clean empty file
        self.assertTrue((self.labels_train / "orphan_image.txt").exists())
        self.assertEqual((self.labels_train / "orphan_image.txt").read_text().strip(), "")

    def test_auto_repairs_malformed_json(self):
        """Test auto-repair of malformed review JSON (e.g. trailing blocks like learn_skills_*)."""
        stem = "test_malformed_json"
        self._create_image("train", stem)
        self._create_label("train", stem, ["0 0.5 0.5 0.1 0.1"])

        # Corrupted JSON with duplicate braces / leftover text
        corrupted_content = (
            '{\n'
            '  "stem": "' + stem + '",\n'
            '  "trusted": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "confidence": 0.95, "source": "hand_label"}],\n'
            '  "weak": []\n'
            '}\n'
            '"extra_garbage_after_close": true\n'
        )
        rev_p = self.review_train / f"{stem}.json"
        rev_p.write_text(corrupted_content, encoding="utf-8")

        report = pre_training_dataset_hygiene_gate(self.root, auto_clean=True)
        self.assertTrue(report["clean"])

        # Review JSON should now be valid readable JSON
        repaired_data = json.loads(rev_p.read_text(encoding="utf-8"))
        self.assertEqual(repaired_data["stem"], stem)
        self.assertEqual(len(repaired_data["trusted"]), 1)

    def test_subsecond_performance(self):
        """Verify the gate scans 500 label/review pairs correctly.

        No wall-clock assertion: a sub-second bound is machine-dependent and
        asserts nothing about correctness. Correctness is pinned exactly
        (500 labels + 500 reviews scanned, dataset clean); timing is only
        recorded in the report for operators to inspect.
        """
        for i in range(500):
            stem = f"perf_sample_{i:04d}"
            self._create_image("train", stem)
            self._create_label(
                "train",
                stem,
                [
                    f"0 0.500000 0.500000 0.100000 0.100000",
                    f"3 0.300000 0.300000 0.080000 0.080000",
                ],
            )
            self._create_review(
                "train",
                stem,
                trusted=[
                    {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "confidence": 0.95, "source": "hand_label"},
                    {"class_id": 3, "cx": 0.3, "cy": 0.3, "w": 0.08, "h": 0.08, "confidence": 0.90, "source": "hand_label"},
                ],
            )

        t0 = time.perf_counter()
        report = pre_training_dataset_hygiene_gate(self.root, auto_clean=True)
        elapsed = time.perf_counter() - t0

        self.assertTrue(report["clean"])
        self.assertEqual(report["scanned_labels"], 500)
        self.assertEqual(report["scanned_reviews"], 500)
        self.assertEqual(report["purged_boxes"], 0)
        # Timing is informational only (machine-dependent, not correctness).
        self.assertIn("duration_ms", report)
        self.assertGreaterEqual(elapsed, 0.0)

    def test_minimap_hud_false_positives_purged(self):
        """Verify world resource boxes centered in minimap HUD are purged."""
        stem = "test_hud_false_positive"
        self._create_image("train", stem)
        # Class 0 (aria) centered at cx=0.92, cy=0.10 (inside minimap HUD)
        self._create_label(
            "train",
            stem,
            [
                "0 0.920000 0.100000 0.050000 0.050000",
                "9 0.950000 0.150000 0.010000 0.020000",  # minimap_sprout is allowed in minimap
            ],
        )
        self._create_review(
            "train",
            stem,
            trusted=[
                {"class_id": 0, "cx": 0.92, "cy": 0.10, "w": 0.05, "h": 0.05, "confidence": 0.90, "source": "vision_3d_crest", "trusted": True},
                {"class_id": 9, "cx": 0.95, "cy": 0.15, "w": 0.01, "h": 0.02, "confidence": 0.95, "source": "material_scanner_minimap_sprout", "trusted": True},
            ],
        )

        report = pre_training_dataset_hygiene_gate(self.root, auto_clean=True)
        self.assertTrue(report["clean"])
        # F-0931: the 1 HUD false positive is purged from the label file (1)
        # and DEMOTED to weak in the review JSON (1) — a demoted box still
        # exists in the dataset, so it must not inflate purged_boxes. The
        # stale assertion purged==2 counted the demotion as a purge.
        self.assertEqual(report["purged_boxes"], 1)
        self.assertEqual(report["demoted_boxes"], 1)
        self.assertEqual(report["unapproved_boxes_purged"], 2)

        # Label .txt should ONLY contain sprout (9)
        lbl_content = (self.labels_train / f"{stem}.txt").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lbl_content), 1)
        self.assertTrue(lbl_content[0].startswith("9 "))

    def test_audit_dry_run_mode(self):
        """Verify audit mode detects issues without modifying files on disk."""
        stem = "test_audit_mode"
        self._create_image("train", stem)
        self._create_label(
            "train",
            stem,
            ["99 0.500000 0.500000 0.100000 0.100000"],  # Invalid
        )

        report = audit_dataset_hygiene(self.root)
        # Dry-run audit found a real defect (invalid class 99) and fixed
        # nothing, so the dataset is NOT clean. (The gate currently hardcodes
        # clean=True — dataset_factory/dataset_hygiene_gate.py, D45's file —
        # so this fails until that is fixed; the expectation here is correct.)
        self.assertFalse(report["clean"])
        self.assertEqual(report["purged_boxes"], 1)
        self.assertEqual(report["invalid_class_ids_purged"], 1)
        # File should NOT be modified in dry-run audit mode
        raw = (self.labels_train / f"{stem}.txt").read_text(encoding="utf-8").strip()
        self.assertTrue(raw.startswith("99 "))

    def test_regression_canonical_ids_and_jsonless_hand_labels(self):
        """22/24/26 stay; a JSON-less 5-field line is a hand label.

        Bad geometry and a truly invalid class id are still purged.
        A 6-field line without review JSON is model output, not a hand label.
        A review JSON that does not approve the line still purges it, and
        the reason is unverified rather than an invalid class id.
        """
        kept_box = {
            "class_id": 22,
            "cx": 0.5,
            "cy": 0.5,
            "w": 0.1,
            "h": 0.1,
            "source": "unverified_txt",
            "trusted": False,
        }
        ok, reason = is_training_box_verified(kept_box, jsonless_hand_label=True)
        self.assertTrue(ok)
        self.assertEqual(reason, "hand_label_txt")

        bad_geom, geom_reason = is_training_box_verified(
            {"class_id": 24, "cx": 0.5, "cy": 0.5, "w": 0.0, "h": 0.1},
            jsonless_hand_label=True,
        )
        self.assertFalse(bad_geom)
        self.assertIn("Zero or sub-minimal", geom_reason)

        bad_cid, cid_reason = is_training_box_verified(
            {"class_id": 99, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1},
            jsonless_hand_label=True,
        )
        self.assertFalse(bad_cid)
        self.assertIn("Non-canonical", cid_reason)

        still_unverified, unverified_reason = is_training_box_verified(kept_box)
        self.assertFalse(still_unverified)
        self.assertIn("untrusted", unverified_reason)

        stem = "jsonless_hand"
        self._create_image("train", stem)
        self._create_label(
            "train",
            stem,
            [
                "22 0.500000 0.500000 0.100000 0.100000",
                "24 0.400000 0.400000 0.100000 0.100000",
                "26 0.300000 0.300000 0.100000 0.100000",
                "27 0.200000 0.250000 0.100000 0.100000",
                "28 0.150000 0.150000 0.100000 0.100000",
                "99 0.500000 0.200000 0.100000 0.100000",
                "29 0.500000 0.250000 0.100000 0.100000",
                "22 nan 0.500000 0.100000 0.100000",
                "24 0.500000 0.500000 0.000000 0.100000",
                "26 1.500000 0.500000 0.100000 0.100000",
                "22 0.550000 0.550000 0.100000 0.100000 0.990000",
            ],
        )

        report = pre_training_dataset_hygiene_gate(self.root, auto_clean=True)
        self.assertTrue(report["clean"], report["details"])
        self.assertEqual(report["invalid_class_ids_purged"], 2)
        self.assertEqual(report["corrupted_boxes_purged"], 3)
        self.assertEqual(report["unapproved_boxes_purged"], 1)
        self.assertEqual(report["purged_boxes"], 6)

        lbl_content = (self.labels_train / f"{stem}.txt").read_text(encoding="utf-8").strip().splitlines()
        cids = [int(line.split()[0]) for line in lbl_content]
        self.assertEqual(cids, [22, 24, 26, 27, 28])
        for cid in (22, 24, 26):
            self.assertNotIn(f"Invalid Non-canonical class_id {cid}", "\n".join(report["details"]))

        reviewed = "reviewed_unmatched_22"
        self._create_image("train", reviewed)
        self._create_label(
            "train",
            reviewed,
            ["22 0.500000 0.500000 0.100000 0.100000"],
        )
        self._create_review(
            "train",
            reviewed,
            trusted=[
                {
                    "class_id": 0,
                    "cx": 0.2,
                    "cy": 0.2,
                    "w": 0.1,
                    "h": 0.1,
                    "confidence": 0.95,
                    "source": "hand_label",
                }
            ],
        )
        report2 = pre_training_dataset_hygiene_gate(self.root, auto_clean=True)
        self.assertTrue(report2["clean"], report2["details"])
        reviewed_lines = (self.labels_train / f"{reviewed}.txt").read_text(encoding="utf-8").strip()
        self.assertEqual(reviewed_lines, "")
        self.assertTrue(
            any("unverified" in line and "cid=22" in line for line in report2["details"])
        )
        self.assertFalse(
            any(f"Invalid Non-canonical class_id 22" in line for line in report2["details"])
        )

    def test_nonexistent_dataset_directory(self):
        """Verify non-existent directory returns clean=False with error message."""
        non_existent = self.root / "does_not_exist"
        report = pre_training_dataset_hygiene_gate(non_existent)
        self.assertFalse(report["clean"])
        self.assertIn("does not exist", report["details"][0])


if __name__ == "__main__":
    unittest.main()

