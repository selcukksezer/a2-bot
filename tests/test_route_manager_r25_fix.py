# -*- coding: utf-8 -*-
"""R25 regression tests.

F-1772: locate_node_on_open_map must NOT try to measure camera yaw from
the open-map frame (there is no minimap player arrow in it); the production
path normalizes the template explicitly with camera_yaw_deg=0.0 because the
open map is fixed-north (north-up).

S-1302: validate_route_manager_contract() (and the 8 functions chaining it)
must reflect real readiness (_real_readiness: default base_dir exists)
instead of the constant ``bool(get_route_manager_capabilities())``.
"""

import os

import cv2
import numpy as np
import pytest

from core.route_manager import (
    NodeData,
    RouteManager,
    get_route_manager_audit_summary,
    get_route_manager_compliance_certificate,
    get_route_manager_integrity_seal,
    validate_route_manager_contract,
)


class _YawRecordingGuard:
    """Stand-in for MinimapOrientationGuard: records how it was called."""

    def __init__(self) -> None:
        self.with_frame_calls = 0
        self.normalize_calls = []

    def normalize_patch_to_north_with_frame(self, patch_bgr, frame_bgr, target_size=None):
        self.with_frame_calls += 1
        raise AssertionError("with_frame must not be called on an open-map frame")

    def normalize_patch_to_north(self, patch_bgr, camera_yaw_deg=0.0, target_size=None):
        self.normalize_calls.append(float(camera_yaw_deg))
        return np.array(patch_bgr, copy=True)


def test_open_map_never_measures_yaw_from_frame(tmp_path):
    """F-1772: Sobel branch on an open-map frame normalizes with yaw 0.0
    explicitly and never calls normalize_patch_to_north_with_frame."""
    base_dir = tmp_path / "routes"
    patch_dir = base_dir / "PilgrimsRidge" / "patches"
    patch_dir.mkdir(parents=True)
    patch_path = patch_dir / "node_1.png"
    rng = np.random.default_rng(7)
    patch = rng.integers(0, 256, size=(80, 80, 3), dtype=np.uint8)
    assert cv2.imwrite(str(patch_path), patch)

    guard = _YawRecordingGuard()
    manager = RouteManager(base_dir=str(base_dir), orientation_guard=guard)
    node = NodeData(id=1, region="PilgrimsRidge", patch_path=str(patch_path),
                    norm_x=0.5, norm_y=0.5)
    # Open-map frame with the patch nowhere in it -> raw template score is
    # ~0, so the Sobel+CLAHE secondary-match branch runs.
    open_map = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)

    result = manager.locate_node_on_open_map(node, open_map)

    assert guard.with_frame_calls == 0
    assert guard.normalize_calls, "normalize_patch_to_north was never called"
    assert guard.normalize_calls == [0.0]
    assert result is not None
    assert result.coords is not None


def test_validate_contract_true_when_default_routes_dir_exists():
    """S-1302: from the repo root the default ``routes/`` dir exists, so the
    real probe (and the whole chain built on it) must report ready."""
    assert os.path.isdir("routes")
    assert validate_route_manager_contract() is True
    assert get_route_manager_compliance_certificate()["certified"] is True
    assert get_route_manager_integrity_seal()["sealed"] is True
    assert get_route_manager_audit_summary()["audit_passed"] is True


def test_validate_contract_false_when_default_routes_dir_missing(tmp_path, monkeypatch):
    """S-1302: with no ``routes/`` dir reachable, the probe must report
    NOT ready instead of the old constant True."""
    monkeypatch.chdir(tmp_path)
    assert not os.path.isdir("routes")
    assert validate_route_manager_contract() is False
    assert get_route_manager_compliance_certificate()["certified"] is False
    assert get_route_manager_audit_summary()["audit_passed"] is False
