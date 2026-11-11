# -*- coding: utf-8 -*-
"""R56: production entry points for the two F-0919/F-1009 modules.

F-0919: `dataset_factory.clean_dataset` was only imported from tests; its
production interface is the operator CLI (`python -m
dataset_factory.clean_dataset`). This file pins that CLI (dry-run + move)
so the tool cannot silently rot.

F-1009: `dataset_factory.six_material_discriminator` was referenced NOWHERE
outside its own file — not even by a test. This file is the first test to
import `SixMaterialDiscriminator` directly, and pins its new CLI entry
point (`python -m dataset_factory.six_material_discriminator`).
Full pipeline wiring (secondary vote inside
`core/material_scanner.py::discriminate_3d_material`) is cross-slice (R07)
and needs a taxonomy call first — see the CLI docstring in the module.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset_factory.clean_dataset import purge_non_game_screenshots
from dataset_factory.six_material_discriminator import (
    SixMaterialDiscriminator,
    main as discriminator_main,
)


def _aria_like_patch(seed: int = 7) -> np.ndarray:
    """Synthetic BGR patch inside the herb_aria HSV profile (hue 70-105)."""
    hsv = np.zeros((64, 64, 3), dtype=np.uint8)
    hsv[:, :, 0] = 85
    hsv[:, :, 1] = 120
    hsv[:, :, 2] = 190
    rng = np.random.default_rng(seed)
    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1].astype(int) + rng.integers(-15, 16, (64, 64)), 0, 255
    ).astype(np.uint8)
    hsv[:, :, 2] = np.clip(
        hsv[:, :, 2].astype(int) + rng.integers(-10, 11, (64, 64)), 0, 255
    ).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _grass_like_patch() -> np.ndarray:
    """Flat mid-green patch: environmental clutter, not a material."""
    hsv = np.zeros((64, 64, 3), dtype=np.uint8)
    hsv[:, :, 0] = 50
    hsv[:, :, 1] = 110
    hsv[:, :, 2] = 100
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class TestSixMaterialDiscriminatorDirect:
    def test_discriminate_aria_like_patch_is_valid_material(self) -> None:
        disc = SixMaterialDiscriminator()
        res = disc.discriminate(_aria_like_patch())
        assert res.predicted_class == "herb_aria"
        assert res.is_valid_material is True
        assert res.is_noise is False
        assert res.confidence >= 0.45

    def test_discriminate_grass_like_patch_is_noise(self) -> None:
        disc = SixMaterialDiscriminator()
        res = disc.discriminate(_grass_like_patch())
        assert res.is_valid_material is False
        assert res.is_noise is True

    def test_filter_detections_keeps_material_drops_noise(self) -> None:
        disc = SixMaterialDiscriminator()
        aria = _aria_like_patch()
        kept = disc.filter_detections(
            aria, [{"bbox": (0, 0, 64, 64), "label": "candidate"}]
        )
        assert len(kept) == 1
        assert kept[0]["material_class"] == "herb_aria"
        assert kept[0]["material_confidence"] >= 0.45
        grass = _grass_like_patch()
        assert disc.filter_detections(grass, [{"bbox": (0, 0, 64, 64)}]) == []

    def test_filter_detections_skips_malformed_bbox(self) -> None:
        disc = SixMaterialDiscriminator()
        kept = disc.filter_detections(
            _aria_like_patch(), [{"label": "no-bbox"}, {"bbox": (0, 0, 64)}]
        )
        assert kept == []


class TestSixMaterialDiscriminatorCli:
    def test_cli_valid_material_exits_zero(self, tmp_path: Path) -> None:
        img = tmp_path / "aria.png"
        assert cv2.imwrite(str(img), _aria_like_patch())
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "dataset_factory.six_material_discriminator",
                "--image",
                str(img),
            ],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert "predicted_class=herb_aria" in proc.stdout

    def test_cli_noise_exits_one_and_bbox_error_exits_two(
        self, tmp_path: Path
    ) -> None:
        img = tmp_path / "grass.png"
        assert cv2.imwrite(str(img), _grass_like_patch())
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "dataset_factory.six_material_discriminator",
                "--image",
                str(img),
            ],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "is_noise=True" in proc.stdout
        assert discriminator_main(["--image", str(img), "--bbox", "1,2,3"]) == 2
        assert (
            discriminator_main(["--image", str(tmp_path / "missing.png")]) == 2
        )


class TestCleanDatasetCli:
    def test_cli_dry_run_reports_invalid_and_move_quarantines(
        self, tmp_path: Path
    ) -> None:
        train_dir = tmp_path / "images" / "train"
        train_dir.mkdir(parents=True)
        bad = train_dir / "desktop.png"
        # Solid black frame: not a valid game frame -> purged/quarantined.
        assert cv2.imwrite(str(bad), np.zeros((32, 32, 3), dtype=np.uint8))

        dry = purge_non_game_screenshots(
            dataset_dir=tmp_path, dry_run=True, verbose=False
        )
        assert dry["total_scanned"] == 1
        assert dry["invalid_purged"] == 1
        assert bad.exists()  # dry-run must not touch the disk

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "dataset_factory.clean_dataset",
                "--dataset-dir",
                str(tmp_path),
                "--action",
                "move",
                "--quiet",
            ],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert not bad.exists()
        assert (tmp_path / "quarantine" / "images" / "train" / "desktop.png").exists()
