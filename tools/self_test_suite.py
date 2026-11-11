# -*- coding: utf-8 -*-
"""
Comprehensive Self-Test & Verification Suite (Gatherion A2 Bot).
Dynamically inspects and tests all modules in config/, core/, ui/, tools/,
verifies template assets in templates/ and assets/, executes OpenCV image
processing pipelines against video_frames/frame_00s.jpg, frame_18s.jpg,
frame_24s.jpg, frame_33s.jpg, and frame_36s.jpg, and outputs a detailed
PASS / FAIL verification table.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

import cv2
import numpy as np
from version import __version__

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NOTE (F-1275): this module must NOT chdir at import time. All paths used
# by the suite are absolute (rooted at PROJECT_ROOT / self.root), so no
# working-directory side effect is needed and importing the suite must not
# move the caller's cwd.

logger = logging.getLogger("Gatherion.SelfTestSuite")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


@dataclass
class SingleTestResult:
    """Represents the outcome of an individual verification check."""

    category: str
    name: str
    passed: bool
    duration_ms: float
    details: str


# Every category string passed to SelfTestSuite._run_check. Kept in one
# place so `--only`/`--list` and the built-in self-test can enumerate the
# suite's real check set instead of a second, drifting list.
CHECK_CATEGORIES: FrozenSet[str] = frozenset(
    {
        "DYNAMIC_IMPORT",
        "MODULES",
        "TEMPLATES",
        "VIDEO_FRAMES",
        "VISION_FRAME_TEST",
        "SUBSYSTEM_CONFIG",
        "SUBSYSTEM_ROUTES",
        "SUBSYSTEM_TELEMETRY",
        "SUBSYSTEM_WATCHDOG",
        "SUBSYSTEM_SUPPORT",
        "SUBSYSTEM_CORE",
        "SUBSYSTEM_UI",
        "PHASE8_8.1",
        "PHASE8_8.2",
        "PHASE8_8.3",
        "PHASE8_8.4",
        "PHASE8_8.5",
        "PHASE8_8.6",
        "PHASE8_8.7",
    }
)


def _safe_imread(path: Path, flags: int) -> Optional[np.ndarray]:
    """Shared decoder behind safe_imread_color / safe_imread_gray.

    D72: the two helpers were byte-for-byte identical except for the
    cv2 flag (22 duplicated lines). Behaviour is unchanged: missing /
    empty / undecodable files return None, any exception is logged.
    """
    try:
        if not path.exists():
            return None
        raw = np.fromfile(str(path), dtype=np.uint8)
        if raw.size == 0:
            return None
        img = cv2.imdecode(raw, flags)
        return img
    except Exception as exc:
        logger.error("safe_imread failed for %s: %s", path, exc)
        return None


def _analyze_ast(rel_path: str, source: str, tree: ast.AST) -> Tuple[int, List[str]]:
    """One ast.walk pass: counts nodes AND detects placeholder stubs.

    D72: run_dynamic_module_checks already walked every module tree to
    count nodes, and the PHASE8_8.7 gate then re-parsed + re-walked the
    same files (27.7s parse + 7.2s walk for 110 files / ~7 MB on this
    box). Both checks now share this single pass; per-file results are
    memoized in SelfTestSuite._ast_placeholder_cache, so the gate reuses
    the dynamic phase's work and only parses the files the dynamic
    phase skips (__init__.py, _-prefixed, main.py). The placeholder
    rules below are the old gate's rules (F-1154 family)
    plus the S-4702 literal-claim tightening:
    single `pass` / `...` bodies, single constant `return`s that claim
    readiness/audit/certification, and TODO/FIXME comments.
    """
    todo_marker = "# TODO"
    fixme_marker = "# FIXME"
    violations: List[str] = []
    node_count = 0
    func_nodes: List[ast.AST] = []
    for node in ast.walk(tree):
        node_count += 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_nodes.append(node)

    for node in func_nodes:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        # Strip leading docstring expression if present
        stmts = list(node.body)
        if (
            stmts
            and isinstance(stmts[0], ast.Expr)
            and isinstance(stmts[0].value, ast.Constant)
            and isinstance(stmts[0].value.value, str)
        ):
            stmts = stmts[1:]

        if len(stmts) == 1:
            only_stmt = stmts[0]
            if isinstance(only_stmt, ast.Pass):
                violations.append(f"{rel_path}:{node.name}(line {node.lineno} empty pass)")
            elif (
                isinstance(only_stmt, ast.Expr)
                and isinstance(only_stmt.value, ast.Constant)
                and only_stmt.value.value is Ellipsis
            ):
                violations.append(f"{rel_path}:{node.name}(line {node.lineno} ellipsis)")
            elif isinstance(only_stmt, ast.Return):
                # F-1154: the old gate only caught `pass` / `...`, so a
                # whole family of single-return stub helpers slipped
                # through — e.g. this file's own get_self_test_suite_*
                # helpers returning {"ready": True} / {"certified": True}
                # without running any test. A function whose entire body
                # is one `return` of a constant (or a dict/list of only
                # constants) that CLAIMS readiness/audit/certification is
                # a placeholder, not an implementation.
                ret = only_stmt.value
                claim_keys = {
                    "ready", "audit_passed", "certified",
                    "sealed", "operational", "alive",
                    "contract_valid", "telemetry_ready",
                    "self_test_ready", "audit", "passed",
                }
                name_hints = (
                    "ready", "audit", "certif", "seal",
                    "contract", "summary", "snapshot",
                    "digest", "telemetry", "health",
                    "heartbeat", "probe",
                )
                lname = node.name.lower()
                if (
                    isinstance(ret, ast.Constant)
                    and isinstance(ret.value, bool)
                    and any(h in lname for h in name_hints)
                ):
                    violations.append(
                        f"{rel_path}:{node.name}(line {node.lineno} stub: returns bare {ret.value} without checking anything)"
                    )
                elif isinstance(ret, ast.Dict) and ret.values:
                    # S-4702: the old gate only flagged a dict
                    # when EVERY value was a constant, so a
                    # mixed dict escaped even when it carried
                    # a literal readiness/audit/certification
                    # claim -- e.g. {"audit_passed": True,
                    # "readiness": get_...()}. A claim key
                    # whose value is a literal constant is a
                    # placeholder claim whether or not the
                    # sibling values are computed: a Call
                    # sibling is NOT evidence the claim was
                    # checked (the callee may itself be a
                    # tautology -- wiring that callee is the
                    # owning slice's stub fix, e.g. S-4701).
                    # Legit helpers stay clean: their claim
                    # values are Calls to a real probe
                    # (validate_*_contract()), not constants.
                    literal_claims = sorted(
                        {
                            k.value
                            for k, v in zip(ret.keys, ret.values)
                            if isinstance(k, ast.Constant)
                            and isinstance(k.value, str)
                            and k.value in claim_keys
                            and isinstance(v, ast.Constant)
                        }
                    )
                    if literal_claims:
                        violations.append(
                            f"{rel_path}:{node.name}(line {node.lineno} stub: returns dict with literal {literal_claims} claim)"
                        )

    for lineno, raw_line in enumerate(source.splitlines(), start=1):
        if todo_marker in raw_line or fixme_marker in raw_line:
            violations.append(f"{rel_path}:L{lineno}(placeholder comment)")

    return node_count, violations


def safe_imread_color(path: Path) -> Optional[np.ndarray]:
    """Reads a BGR image safely across Windows paths containing non-ASCII characters."""
    return _safe_imread(path, cv2.IMREAD_COLOR)


def safe_imread_gray(path: Path) -> Optional[np.ndarray]:
    """Reads a grayscale image safely across Windows paths containing non-ASCII characters."""
    return _safe_imread(path, cv2.IMREAD_GRAYSCALE)


def safe_imwrite(path: Path, img: np.ndarray) -> bool:
    """Writes an image safely across Windows paths containing non-ASCII characters."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix if path.suffix else ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception as exc:
        logger.error("safe_imwrite failed for %s: %s", path, exc)
        return False


class SelfTestSuite:
    """
    Full-coverage automated verification suite for Gatherion A2 Bot.
    Dynamically validates config, core, ui, tools, templates, and video_frames.
    """

    REQUIRED_VIDEO_FRAMES: Tuple[str, str, str, str, str] = (
        "video_frames/frame_00s.jpg",
        "video_frames/frame_18s.jpg",
        "video_frames/frame_24s.jpg",
        "video_frames/frame_33s.jpg",
        "video_frames/frame_36s.jpg",
    )

    PACKAGE_DIRS: Tuple[str, str, str, str] = (
        "config",
        "core",
        "ui",
        "tools",
    )

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root: Path = Path(root) if root else PROJECT_ROOT
        self.results: List[SingleTestResult] = []
        self._qt_app: Any = None
        # D72: per-run caches. _img_cache dedups JPEG/PNG decodes of the
        # same file (up to 6x per run for pilgrims_ridge_1.jpg, 5x for
        # map_header.png). _ast_placeholder_cache memoizes the shared
        # AST placeholder scan so the 8.7 gate reuses the dynamic phase's
        # parse+walk. Keys carry size+mtime so an on-disk change is
        # never served stale. _only_categories is set only by run_all()
        # and reset afterwards, so direct run_* calls always run full.
        self._img_cache: Dict[str, np.ndarray] = {}
        self._ast_placeholder_cache: Dict[str, List[str]] = {}
        self._only_categories: Optional[FrozenSet[str]] = None

    def _ensure_qt_app(self) -> Any:
        """Initializes a shared QApplication instance for UI module verification."""
        if self._qt_app is not None:
            return self._qt_app
        try:
            from PyQt6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv if sys.argv else ["self_test_suite"])
            self._qt_app = app
            return self._qt_app
        except Exception as exc:
            logger.warning("QApplication initialization warning: %s", exc)
            return None

    def _imread_cached(self, path: Path, gray: bool = False) -> Optional[np.ndarray]:
        """Decodes an image once per suite run and reuses the result.

        Returns a COPY each time, so callers may crop/draw on the array
        without aliasing each other's pixels (e.g. the same pilgrims
        frame is handed to RouteManager, AutoCartographer, MaterialScanner,
        VisionEngine, GameHUDReader and CombatRecovery). Failed decodes
        are not cached, preserving the old re-check behaviour.
        """
        fpath = Path(path)
        try:
            st = fpath.stat()
            key = f"{fpath.resolve()}|{st.st_size}|{st.st_mtime_ns}|{'gray' if gray else 'color'}"
        except OSError:
            key = f"{fpath}|{'gray' if gray else 'color'}"
        hit = self._img_cache.get(key)
        if hit is not None:
            return hit.copy()
        img = safe_imread_gray(fpath) if gray else safe_imread_color(fpath)
        if img is not None:
            self._img_cache[key] = img
            return img.copy()
        return img

    def _ast_cache_key(self, fpath: Path) -> str:
        """Cache key for the per-file AST placeholder scan (path+size+mtime)."""
        try:
            st = Path(fpath).stat()
            return f"{Path(fpath).resolve()}|{st.st_size}|{st.st_mtime_ns}"
        except OSError:
            return str(fpath)

    def _scan_repo_placeholders(self) -> Tuple[List[str], int]:
        """Runs the full AST zero-placeholder gate scan over the repo.

        Reuses per-file results memoized by run_dynamic_module_checks
        (same _analyze_ast pass); files the dynamic phase skips
        (__init__.py, _-prefixed, main.py) are parsed + scanned here.
        Calling this without a prior dynamic phase parses everything,
        exactly like the old gate did. Returns (violations, files_scanned).
        """
        root_str = str(self.root)
        scan_dirs = ("config", "core", "ui", "tools")
        py_files: List[str] = []
        for dname in scan_dirs:
            full_dir = os.path.join(root_str, dname)
            if not os.path.isdir(full_dir):
                continue
            for entry in sorted(os.listdir(full_dir)):
                if entry.endswith(".py"):
                    py_files.append(os.path.join(full_dir, entry))

        main_py = os.path.join(root_str, "main.py")
        if os.path.isfile(main_py):
            py_files.append(main_py)

        violations: List[str] = []
        for fpath in py_files:
            rel_path = os.path.relpath(fpath, root_str)
            key = self._ast_cache_key(Path(fpath))
            cached = self._ast_placeholder_cache.get(key)
            if cached is None:
                with open(fpath, "r", encoding="utf-8") as f:
                    source = f.read()
                try:
                    tree = ast.parse(source, filename=rel_path)
                except SyntaxError as exc:
                    violations.append(f"{rel_path}:SyntaxError(line {exc.lineno})")
                    continue
                _, cached = _analyze_ast(rel_path, source, tree)
                self._ast_placeholder_cache[key] = cached
            violations.extend(cached)

        return violations, len(py_files)

    def _run_check(
        self,
        category: str,
        name: str,
        fn: Callable[[], Tuple[bool, str]],
    ) -> Optional[SingleTestResult]:
        """Executes a test callable, measures duration, catches exceptions, and records the result.

        When run_all() was given --only categories, checks outside the
        selection are NOT executed and NOT recorded; the filter banner
        printed by run_all() says so explicitly (never a silent skip).
        """
        if self._only_categories is not None and category not in self._only_categories:
            return None
        t0 = time.perf_counter()
        try:
            ok, details = fn()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            res = SingleTestResult(
                category=category,
                name=name,
                passed=bool(ok),
                duration_ms=round(elapsed_ms, 2),
                details=str(details),
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.error("Test [%s] %s raised exception: %s", category, name, exc)
            res = SingleTestResult(
                category=category,
                name=name,
                passed=False,
                duration_ms=round(elapsed_ms, 2),
                details=f"Exception: {exc}",
            )
        self.results.append(res)
        return res

    # =========================================================================
    # 1. Dynamic Module Discovery, AST Syntax & Import Checks
    # =========================================================================

    def run_dynamic_module_checks(self) -> None:
        """Dynamically scans config/, core/, ui/, and tools/ and verifies AST syntax and imports."""
        self._ensure_qt_app()

        for pkg in self.PACKAGE_DIRS:
            pkg_dir = self.root / pkg
            if not pkg_dir.exists():
                self._run_check(
                    "MODULES",
                    f"Directory {pkg}/",
                    lambda p=pkg: (False, f"Directory '{p}' does not exist"),
                )
                continue

            py_files = sorted(
                f
                for f in pkg_dir.glob("*.py")
                if f.name != "__init__.py" and not f.name.startswith("_")
            )
            for py_file in py_files:
                mod_name = f"{pkg}.{py_file.stem}"

                def _check_mod(fpath: Path = py_file, mname: str = mod_name) -> Tuple[bool, str]:
                    source = fpath.read_text(encoding="utf-8", errors="replace")
                    tree = ast.parse(source, filename=str(fpath))
                    # D72: this one walk also runs the placeholder rules,
                    # and the result is memoized for the 8.7 gate so the
                    # repo is not parsed + walked a second time.
                    node_count, violations = _analyze_ast(
                        f"{pkg}/{py_file.name}", source, tree
                    )
                    self._ast_placeholder_cache[self._ast_cache_key(fpath)] = violations
                    imported = importlib.import_module(mname)
                    exported = [attr for attr in dir(imported) if not attr.startswith("_")]
                    # F-1155: AST + import + export count alone proves nothing.
                    # Require real, module-owned behaviour: at least one public
                    # callable, and the imported file must be the scanned file
                    # (no sys.path shadowing). A module that only re-exports
                    # (e.g. ui.overlay_window) passes with an explicit note.
                    own_callables = [
                        attr
                        for attr in exported
                        if callable(getattr(imported, attr))
                        and getattr(getattr(imported, attr), "__module__", None) == mname
                    ]
                    any_callable = any(
                        callable(getattr(imported, attr)) for attr in exported
                    )
                    mod_file = getattr(imported, "__file__", "") or ""
                    file_match = bool(mod_file) and Path(mod_file).resolve() == fpath.resolve()
                    if node_count == 0 or not exported or not any_callable or not file_match:
                        return (
                            False,
                            f"AST nodes={node_count}, exports={len(exported)}, "
                            f"own_callables={len(own_callables)}, "
                            f"file_match={file_match} (imported={mod_file or '?'})",
                        )
                    note = "re-export-only" if not own_callables else "ok"
                    return (
                        True,
                        f"AST nodes={node_count}, exports={len(exported)} symbols, "
                        f"own_callables={len(own_callables)}, file_match={file_match} ({note})",
                    )

                self._run_check("DYNAMIC_IMPORT", mod_name, _check_mod)

    # =========================================================================
    # 2. Templates & Video Frames Inventory Checks
    # =========================================================================

    def _verify_templates(self) -> Tuple[bool, str]:
        """Verifies the required template images against real references.

        D72: two changes, verdict semantics unchanged. (1) Early exit:
        the gate is ">= 0.75 against at least one reference", so once a
        reference qualifies the verdict is fixed; scanning the remaining
        (expensive, full-size) references can only raise the reported
        score, never change pass/fail — the loop stops and reports the
        FIRST qualifying match honestly as such. The failure branch
        still scans every reference and reports the true best score.
        (2) Reference/template images go through _imread_cached instead
        of decoding the same JPEG once per template.
        """
        # F-1125: NEVER generate templates from the very frames under
        # test (frame_33s/frame_18s). Cutting a template out of a frame
        # and then "detecting" it in the same frame is a circular
        # success. Templates are independent repo assets; if one is
        # missing the suite must FAIL and say which one, not silently
        # manufacture it from the test data.
        templates_dir = self.root / "templates"
        required = (
            "f_gather_template.png",
            "f_gather.png",
            "f_gather_pill.png",
            "player_arrow.png",
            "map_header.png",
        )
        missing = [t for t in required if not (templates_dir / t).exists()]
        if missing:
            return False, f"Template missing (not generated from test frames): {missing}"

        # F-1220: "readable and non-empty" is not validation. Require a
        # real image (minimum size, non-uniform pixels) AND require the
        # template to actually match some real reference image. The
        # templates come from heterogeneous sources (video frames and
        # reference screenshots at different resolutions), so no single
        # designated frame is valid for all of them: each template must
        # score >= 0.75 against at least one video frame or reference
        # screenshot, and the report names which one matched.
        ref_images: List[Tuple[str, np.ndarray]] = []
        for ref_path in sorted((self.root / "video_frames").glob("frame_*.jpg")):
            ref_img = self._imread_cached(ref_path)
            if ref_img is not None:
                ref_images.append((ref_path.name, cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)))
        for ref_path in sorted((self.root / "reference_screenshots").glob("*.jpg")):
            ref_img = self._imread_cached(ref_path)
            if ref_img is not None:
                ref_images.append((ref_path.name, cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)))
        if not ref_images:
            return False, "No reference frames/screenshots available to validate templates against"
        verified: List[str] = []
        for tname in required:
            tpath = templates_dir / tname
            img = self._imread_cached(tpath)
            if img is None or img.size == 0:
                return False, f"Template missing or unreadable: {tname}"
            h, w = img.shape[:2]
            if w < 8 or h < 8:
                return False, f"Template implausibly small: {tname} ({w}x{h})"
            if float(np.std(img)) <= 1.0:
                return False, f"Template is blank/uniform: {tname} (std<=1.0)"
            gray_tmpl = (
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            )
            best_score = 0.0
            best_ref = ""
            for ref_name, gray_ref in ref_images:
                if gray_ref.shape[0] < gray_tmpl.shape[0] or gray_ref.shape[1] < gray_tmpl.shape[1]:
                    continue
                res = cv2.matchTemplate(gray_ref, gray_tmpl, cv2.TM_CCOEFF_NORMED)
                score = float(res.max())
                if score > best_score:
                    best_score, best_ref = score, ref_name
                # D72 early exit (see method docstring): the first
                # reference at >= 0.75 settles the verdict.
                if best_score >= 0.75:
                    break
            if best_score < 0.75:
                return (
                    False,
                    f"Template matches no real reference image: {tname} "
                    f"(best score={best_score:.3f} < 0.75 across {len(ref_images)} references)",
                )
            verified.append(f"{tname}({w}x{h}, match={best_score:.2f} vs {best_ref} [first >= 0.75])")

        return True, ", ".join(verified)

    def run_asset_and_template_checks(self) -> None:
        """Ensures templates are extracted/available and verifies all required template images."""

        self._run_check("TEMPLATES", "Verify templates/ directory & images", self._verify_templates)

        for rel_frame in self.REQUIRED_VIDEO_FRAMES:
            def _check_frame(rf: str = rel_frame) -> Tuple[bool, str]:
                fpath = self.root / rf
                img = self._imread_cached(fpath)
                if img is None or img.size == 0:
                    return False, f"Frame missing or corrupt: {rf}"
                h, w, c = img.shape
                # F-1221: "it loaded" is not a test. Require plausible
                # dimensions, a non-trivial brightness range and real
                # content (a black/blank frame must fail).
                mean_val = float(np.mean(img))
                std_val = float(np.std(img))
                if w < 640 or h < 360:
                    return False, f"Frame implausibly small: {rf} ({w}x{h})"
                if not (10.0 <= mean_val <= 245.0):
                    return False, f"Frame brightness out of range: {rf} (mean_lum={mean_val:.1f})"
                if std_val <= 5.0:
                    return False, f"Frame is blank/uniform: {rf} (std={std_val:.1f})"
                return True, f"Loaded {w}x{h}x{c} (mean_lum={mean_val:.1f}, std={std_val:.1f})"

            self._run_check("VIDEO_FRAMES", f"Frame {rel_frame}", _check_frame)

    # =========================================================================
    # 3. Vision & OpenCV Processing Tests on Required Reference Frames
    # =========================================================================

    def run_video_frame_vision_tests(self) -> None:
        """
        Executes OpenCV image processing and template detection tests against:
        - video_frames/frame_00s.jpg (World view, map closed, no gather prompt)
        - video_frames/frame_18s.jpg (World Map open, map header & player arrow match)
        - video_frames/frame_24s.jpg (World Map open with route pins & patch matching)
        - video_frames/frame_33s.jpg (Active 'F Gather' prompt detection)
        - video_frames/frame_36s.jpg (Post-gather world state verification)
        """
        templates_dir = self.root / "templates"

        def _is_map_open_in_frame(frame_bgr: np.ndarray) -> Tuple[bool, float, float, bool]:
            """Evaluates map open state using map_header template score and center HSV saturation.

            F-1222: the two signals must AGREE. The old
            ``header >= 0.70 or sat < 52`` let a low-saturation frame pass
            as "map open" even when the header template did not match at
            all, masking a header-detection failure. Now the header match
            is the decision (``>= 0.70``); saturation is corroboration
            only. Disagreement returns ``ambiguous=True`` and the caller
            must FAIL the check instead of picking the convenient signal.
            """
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            h, w = frame_bgr.shape[:2]
            center_sat = float(
                np.mean(hsv[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75), 1])
            )
            header_score = 0.0
            for hdr_cand in (
                templates_dir / "legacy" / "map_header_video.png",
                templates_dir / "map_header.png",
            ):
                header_tmpl = self._imread_cached(hdr_cand)
                if (
                    header_tmpl is not None
                    and header_tmpl.shape[1] >= 60
                    and header_tmpl.shape[0] <= h
                    and header_tmpl.shape[1] <= w
                ):
                    res = cv2.matchTemplate(frame_bgr, header_tmpl, cv2.TM_CCOEFF_NORMED)
                    header_score = max(header_score, float(res.max()))
            header_open = header_score >= 0.70
            sat_open = center_sat < 52.0
            ambiguous = header_open != sat_open
            return header_open, header_score, center_sat, ambiguous

        def _match_gather_prompt(frame_bgr: np.ndarray) -> Tuple[bool, float, Tuple[int, int]]:
            """Matches the F Gather template in the central-right prompt region of a frame."""
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            rx1, rx2 = int(w * 0.45), int(w * 0.85)
            ry1, ry2 = int(h * 0.45), int(h * 0.85)
            roi = gray[ry1:ry2, rx1:rx2]

            best_score = 0.0
            best_center = (0, 0)
            for tmpl_cand in (
                templates_dir / "f_gather_template.png",
                templates_dir / "legacy" / "f_gather_video.png",
                self.root / "assets" / "f_gather_template.png",
            ):
                tmpl = self._imread_cached(tmpl_cand, gray=True)
                if tmpl is None:
                    continue
                search_img = roi
                ox, oy = rx1, ry1
                if search_img.shape[0] < tmpl.shape[0] or search_img.shape[1] < tmpl.shape[1]:
                    search_img = gray
                    ox, oy = 0, 0
                if search_img.shape[0] < tmpl.shape[0] or search_img.shape[1] < tmpl.shape[1]:
                    continue
                res = cv2.matchTemplate(search_img, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                score = float(max_val)
                if score > best_score:
                    best_score = score
                    cx = ox + int(max_loc[0]) + tmpl.shape[1] // 2
                    cy = oy + int(max_loc[1]) + tmpl.shape[0] // 2
                    best_center = (cx, cy)
            return best_score >= 0.75, best_score, best_center

        # Test 1: frame_00s.jpg (World view, HUD visible, map closed, no gather prompt)
        def _test_frame_00s() -> Tuple[bool, str]:
            img = self._imread_cached(self.root / "video_frames" / "frame_00s.jpg")
            if img is None:
                return False, "Could not load frame_00s.jpg"
            map_open, hdr_score, sat, ambiguous = _is_map_open_in_frame(img)
            prompt_detected, prompt_score, _ = _match_gather_prompt(img)
            ok = (not map_open) and (not prompt_detected) and (not ambiguous)
            return (
                ok,
                f"map_open={map_open} (hdr={hdr_score:.2f}, sat={sat:.1f}, ambiguous={ambiguous}), "
                f"gather_prompt={prompt_detected} (score={prompt_score:.2f})",
            )

        self._run_check(
            "VISION_FRAME_TEST",
            "frame_00s.jpg: World state (map closed, no prompt)",
            _test_frame_00s,
        )

        # Test 2: frame_18s.jpg (World map open, map header & player arrow detection)
        def _test_frame_18s() -> Tuple[bool, str]:
            img = self._imread_cached(self.root / "video_frames" / "frame_18s.jpg")
            if img is None:
                return False, "Could not load frame_18s.jpg"
            map_open, hdr_score, sat, ambiguous = _is_map_open_in_frame(img)
            arrow_score = 0.0
            for arr_cand in (
                templates_dir / "player_arrow.png",
                templates_dir / "legacy" / "player_arrow_video.png",
            ):
                arrow_tmpl = self._imread_cached(arr_cand)
                if arrow_tmpl is not None:
                    res = cv2.matchTemplate(img, arrow_tmpl, cv2.TM_CCOEFF_NORMED)
                    arrow_score = max(arrow_score, float(res.max()))
            ok = map_open and (not ambiguous) and hdr_score >= 0.80 and arrow_score >= 0.80
            return (
                ok,
                f"map_open={map_open}, header_match={hdr_score:.3f}, "
                f"player_arrow_match={arrow_score:.3f}, sat={sat:.1f}, ambiguous={ambiguous}",
            )

        self._run_check(
            "VISION_FRAME_TEST",
            "frame_18s.jpg: World Map open & player arrow match",
            _test_frame_18s,
        )

        # Test 3: frame_24s.jpg (Map pin / 80x80 node patch extraction & localization)
        def _test_frame_24s() -> Tuple[bool, str]:
            from core.route_manager import RouteManager

            img = self._imread_cached(self.root / "video_frames" / "frame_24s.jpg")
            if img is None:
                return False, "Could not load frame_24s.jpg"
            map_open, hdr_score, _, ambiguous = _is_map_open_in_frame(img)
            if ambiguous:
                return (
                    False,
                    f"Map-open signals disagree on frame_24s.jpg: header_match={hdr_score:.2f} "
                    f"(< 0.70, header template does not match this frame) while saturation "
                    f"suggests map open — header detection gap, not a pass",
                )
            if not map_open:
                return False, f"Map not detected open on frame_24s.jpg (hdr={hdr_score:.2f})"

            # P3-6: CWD'ye göreli "_tmp_test_routes_24s" + koşulsuz rmtree
            # kaldırıldı. Geçici dizini tempfile yönetir: test bitince
            # otomatik silinir, üretim/CWD dizinine dokunulmaz.
            with tempfile.TemporaryDirectory(prefix="a2_test_routes_24s_") as tmp_dir:
                rm = RouteManager(base_dir=tmp_dir)
                h, w = img.shape[:2]
                pin_x, pin_y = int(w * 0.72), int(h * 0.42)
                gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                patch_80 = gray_full[pin_y - 40 : pin_y + 40, pin_x - 40 : pin_x + 40].copy()
                node = rm.add_node(
                    region="Altgard",
                    route="Dredgion Crash Site",
                    patch_img=patch_80,
                    map_x=pin_x,
                    map_y=pin_y,
                    norm_x=round(pin_x / float(w), 4),
                    norm_y=round(pin_y / float(h), 4),
                )
                if node is None:
                    return False, "add_node returned None on frame_24s.jpg"
                # F-1156: locating a patch cut from this very frame, in this
                # very frame, from the in-memory node is a tautology (score
                # 1.0 is guaranteed). Locate instead via a FRESH
                # RouteManager that reloads the node + patch from disk, and
                # require a real template match (used_fallback must be
                # False): a coordinate fallback would return the recorded
                # pin position even if matching failed.
                rm_disk = RouteManager(base_dir=tmp_dir)
                node_disk = rm_disk.get_node(int(node.id), region="Altgard")
                if node_disk is None:
                    return False, "Node was not persisted to disk by add_node"
                loc = rm_disk.locate_node_on_open_map(node_disk, img, match_threshold=0.75)
                if loc is None:
                    return False, "locate_node_on_open_map failed on frame_24s.jpg"
                (lx, ly), conf, dist_str = loc
                if bool(getattr(loc, "used_fallback", False)):
                    return (
                        False,
                        f"locate_node_on_open_map fell back to recorded coords "
                        f"(conf={conf:.3f}) instead of template-matching the persisted patch",
                    )
                err_px = abs(lx - pin_x) + abs(ly - pin_y)
                if conf < 0.90 or err_px > 4:
                    return (
                        False,
                        f"Persisted-patch match too weak: conf={conf:.3f} "
                        f"at ({lx},{ly}), err={err_px}px",
                    )
                # Negative control: the same persisted patch must NOT match
                # confidently on an unrelated world-view frame (frame_00s,
                # map closed). If it does, the matcher cannot discriminate.
                frame00 = self._imread_cached(self.root / "video_frames" / "frame_00s.jpg")
                if frame00 is not None:
                    neg = rm_disk.locate_node_on_open_map(node_disk, frame00, match_threshold=0.75)
                    if neg is not None and (not bool(getattr(neg, "used_fallback", True))) and float(neg.confidence) >= 0.75:
                        return (
                            False,
                            f"Persisted patch false-matched on unrelated frame_00s.jpg "
                            f"(conf={float(neg.confidence):.3f}) — matcher does not discriminate",
                        )
                return (
                    True,
                    f"map_open={map_open} (hdr={hdr_score:.2f}), persisted-patch match={conf:.3f} "
                    f"at ({lx},{ly}), dist={dist_str}, negative-control(frame_00s)=clean",
                )

        self._run_check(
            "VISION_FRAME_TEST",
            "frame_24s.jpg: Map node 80x80 patch record & match",
            _test_frame_24s,
        )

        # Test 4: frame_33s.jpg (Active 'F Gather' prompt detection)
        def _test_frame_33s() -> Tuple[bool, str]:
            img = self._imread_cached(self.root / "video_frames" / "frame_33s.jpg")
            if img is None:
                return False, "Could not load frame_33s.jpg"
            detected, score, (cx, cy) = _match_gather_prompt(img)
            # F-1157 (same threshold as the matcher): _match_gather_prompt
            # reports detected at >= 0.75, so the test must use that same
            # 0.75 bar — not a stricter/looser ad-hoc number.
            ok = detected and score >= 0.75
            return (
                ok,
                f"gather_prompt_detected={detected}, confidence={score * 100.0:.1f}%, "
                f"pos=({cx},{cy})",
            )

        self._run_check(
            "VISION_FRAME_TEST",
            "frame_33s.jpg: 'F Gather' prompt recognition",
            _test_frame_33s,
        )

        # Test 5: frame_36s.jpg (Post-gather / walking state analysis)
        def _test_frame_36s() -> Tuple[bool, str]:
            img = self._imread_cached(self.root / "video_frames" / "frame_36s.jpg")
            if img is None:
                return False, "Could not load frame_36s.jpg"
            map_open, hdr_score, sat, ambiguous = _is_map_open_in_frame(img)
            detected, score, _ = _match_gather_prompt(img)
            # In frame_36s.jpg the map is closed (world view) and gather prompt has disappeared
            # F-1157: the old `score < 0.85` passed while the prompt was
            # DETECTED (matcher threshold is 0.75): a score of 0.80 meant
            # "prompt active" and "test passed" at the same time. Use the
            # matcher's own verdict (`detected`, i.e. >= 0.75) instead.
            ok = (not map_open) and (not ambiguous) and (not detected)
            return (
                ok,
                f"map_open={map_open} (hdr={hdr_score:.2f}, sat={sat:.1f}, ambiguous={ambiguous}), "
                f"prompt_active={detected} (score={score:.2f})",
            )

        self._run_check(
            "VISION_FRAME_TEST",
            "frame_36s.jpg: Post-gather walking transition state",
            _test_frame_36s,
        )

    # =========================================================================
    # 4. Functional Subsystem Unit Tests (Config, Core, Watchdog, UI)
    # =========================================================================

    def run_functional_subsystem_tests(self) -> None:
        """Executes functional unit tests across Config, RouteManager, Telemetry, Watchdog, and UI."""

        # 4.1 Config Schema & SettingsManager test
        def _test_config_schema() -> Tuple[bool, str]:
            from config.schema import SettingsManager

            with tempfile.TemporaryDirectory() as tmp_dir:
                cfg_file = Path(tmp_dir) / "settings.json"
                mgr = SettingsManager(cfg_file)
                mgr.set("gathering", "f_interval_min", 1.75)
                mgr.set("route", "rest_minutes", 8)
                mgr.set("breaks", "break_every_min", 45)
                reloaded = SettingsManager(cfg_file)
                ok = (
                    abs(float(reloaded.get("gathering", "f_interval_min")) - 1.75) < 1e-6
                    and int(reloaded.get("route", "rest_minutes")) == 8
                    and int(reloaded.get("breaks", "break_every_min")) == 45
                )
                return ok, f"Saved & reloaded settings.json ({cfg_file.stat().st_size} bytes)"

        self._run_check("SUBSYSTEM_CONFIG", "SettingsManager CRUD & persistence", _test_config_schema)

        # 4.2 RouteManager CRUD & ZIP Export/Import test
        def _test_route_manager_zip() -> Tuple[bool, str]:
            from core.route_manager import RouteManager

            with tempfile.TemporaryDirectory() as tmp_dir:
                rm = RouteManager(base_dir=os.path.join(tmp_dir, "routes"))
                rm.create_region("Verteron")
                rm.create_route("Verteron", "Ore Loop")
                # F-1223: a flat grey patch proves nothing about the ZIP
                # round-trip (any 80x80 grey block "matches"). Use a
                # deterministic non-uniform pattern and compare the
                # restored patch PIXELS against the original.
                patch = ((np.arange(80, dtype=np.uint16)[:, None] * 3 + np.arange(80, dtype=np.uint16)[None, :] * 7) % 256).astype(np.uint8)
                node = rm.add_node(
                    region="Verteron",
                    routes=["Ore Loop"],
                    patch_img=patch,
                    map_x=640,
                    map_y=360,
                    in_air=True,
                )
                zip_path = os.path.join(tmp_dir, "routes_backup.zip")
                exported_file = rm.export_route_zip("Verteron", "Ore Loop", zip_path)
                rm2 = RouteManager(base_dir=os.path.join(tmp_dir, "routes_restored"))
                imported_reg = rm2.import_route_zip(exported_file)
                restored_nodes = rm2.get_nodes("Verteron", "Ore Loop")
                if len(restored_nodes) != 1:
                    return False, f"ZIP restore returned {len(restored_nodes)} node(s), expected 1"
                restored = restored_nodes[0]
                restored_patch = None
                patch_path = getattr(restored, "patch_path", "") or ""
                if patch_path:
                    cand = patch_path if os.path.isabs(patch_path) else os.path.join(tmp_dir, "routes_restored", patch_path)
                    restored_patch = safe_imread_gray(Path(cand))
                    if restored_patch is None:
                        alt = os.path.join(tmp_dir, "routes_restored", "Verteron", "patches", os.path.basename(patch_path))
                        restored_patch = safe_imread_gray(Path(alt))
                if restored_patch is None:
                    return False, f"Restored node has no readable patch on disk (patch_path={patch_path!r})"
                patch_equal = bool(np.array_equal(restored_patch, patch))
                ok = (
                    os.path.isfile(exported_file)
                    and imported_reg == "Verteron"
                    and restored.id == node.id
                    and patch_equal
                )
                return (
                    ok,
                    f"Exported & restored {len(restored_nodes)} node(s) in '{imported_reg}' via ZIP, "
                    f"patch_pixels_identical={patch_equal}",
                )

        self._run_check("SUBSYSTEM_ROUTES", "RouteManager CRUD & ZIP import/export", _test_route_manager_zip)

        # 4.3 TelemetryEngine metrics & report test
        def _test_telemetry_engine() -> Tuple[bool, str]:
            from core.telemetry import TelemetryEngine

            with tempfile.TemporaryDirectory() as tmp_dir:
                eng = TelemetryEngine()
                eng.record_gather(
                    node_id=1,
                    region="Altgard",
                    route="Dredgion Crash Site",
                    resource_type="Orichalcum Ore",
                    walk_duration_sec=14.2,
                    match_pct=92,
                )
                eng.record_failure(node_id=2, reason="timeout")
                eng.record_combat(duration_sec=5.5, outcome="CLEARED")
                metrics = eng.get_live_metrics()
                json_out = os.path.join(tmp_dir, "telemetry.json")
                csv_out = os.path.join(tmp_dir, "stats.csv")
                saved = eng.save_session_report(json_path=json_out, csv_path=csv_out)
                ok = (
                    saved
                    and metrics["gathered_total"] == 1
                    and metrics["failed_total"] == 1
                    and metrics["combat_count"] == 1
                )
                return (
                    ok,
                    f"gathered={metrics['gathered_total']}, rate={metrics['gathers_per_hour']}/hr, "
                    f"success={metrics['success_rate_pct']}%",
                )

        self._run_check("SUBSYSTEM_TELEMETRY", "TelemetryEngine live metrics & export", _test_telemetry_engine)

        # 4.4 BotWatchdog heartbeat, stall recovery, and display notifications test
        def _test_bot_watchdog() -> Tuple[bool, str]:
            from core.watchdog import BotWatchdog, _query_system_resolution

            recoveries: List[Dict[str, Any]] = []
            display_events: List[str] = []

            # F-1224: the simulation hook alone does not test the real
            # screen. First exercise the REAL display path (no simulation
            # active): the system resolution query must return a sane
            # tuple and is_display_valid() must answer from it.
            wd = BotWatchdog(
                stall_timeout_sec=0.15,
                check_interval_sec=0.05,
                recovery_callback=lambda status: recoveries.append(status),
                display_alert_callback=lambda ev_type, info: display_events.append(ev_type),
            )
            real_res = _query_system_resolution()
            real_res_ok = (
                isinstance(real_res, tuple)
                and len(real_res) == 2
                and all(isinstance(v, int) and v >= 640 for v in real_res)
            )
            if not real_res_ok:
                return False, f"Real display query returned implausible resolution: {real_res!r}"
            real_valid = wd.is_display_valid()
            if not isinstance(real_valid, bool):
                return False, "is_display_valid() did not return a bool on the real display path"
            wd.record_heartbeat(state="WALKING", mode="route", latency_ms=11.0)
            wd.record_heartbeat(state="GATHERING", mode="route", latency_ms=12.0)
            valid_initial = wd.is_display_valid()

            # Simulate stall exceeding 0.15s
            time.sleep(0.22)
            health_stalled = wd.check_health()
            was_stalled = bool(health_stalled.get("stalled"))

            # Simulate resolution change anomaly
            wd.simulate_display_anomaly(resolution=(1024, 768), minimized=False)
            invalid_res = not wd.is_display_valid()

            # Simulate minimized game window anomaly
            wd.reset_display_baseline()
            wd.simulate_display_anomaly(resolution=None, minimized=True)
            invalid_min = not wd.is_display_valid()

            wd.reset_display_baseline()
            wd.record_heartbeat(state="WALKING", mode="route", latency_ms=10.5)
            recovered_clean = not wd.is_stalled() and wd.is_display_valid()
            # After the baseline reset the watchdog must agree with the
            # real system resolution again (no stale simulation left).
            baseline_after_reset = wd.reset_display_baseline()
            baseline_matches_real = tuple(int(v) for v in baseline_after_reset) == tuple(int(v) for v in real_res)

            ok = (
                real_res_ok
                and real_valid
                and valid_initial
                and was_stalled
                and len(recoveries) >= 1
                and invalid_res
                and invalid_min
                and "RESOLUTION_CHANGED" in display_events
                and "WINDOW_MINIMIZED" in display_events
                and recovered_clean
                and baseline_matches_real
            )
            return (
                ok,
                f"real_display={real_res} (valid={real_valid}), stall_detected={was_stalled}, "
                f"recoveries={len(recoveries)}, alerts={display_events}, "
                f"baseline_matches_real={baseline_matches_real}",
            )

        self._run_check("SUBSYSTEM_WATCHDOG", "BotWatchdog stall recovery & display alerts", _test_bot_watchdog)

        # 4.5 SupportDiagnostics report generation & clipboard copy test
        def _test_support_diagnostics() -> Tuple[bool, str]:
            from core.watchdog import SupportDiagnostics

            with tempfile.TemporaryDirectory() as tmp_dir:
                log_file = Path(tmp_dir) / "bot.log"
                log_lines = [f"[12:00:{i:02d}] [INFO] Sample log entry #{i + 1}" for i in range(40)]
                log_file.write_text("\n".join(log_lines), encoding="utf-8")

                diag = SupportDiagnostics(project_root=self.root)
                # The Qt clipboard path needs a QApplication instance to
                # exist BEFORE the copy call; without one the copy silently
                # falls through to Win32/PowerShell fallbacks (which do not
                # exist on Linux) and nothing is ever placed on the
                # clipboard.
                app = self._ensure_qt_app()
                if app is None:
                    return False, "QApplication unavailable — clipboard cannot be verified"
                report = diag.copy_report_to_clipboard(log_path=log_file)
                has_header = "GATHERION A2 BOT - SUPPORT DIAGNOSTIC REPORT" in report
                has_tail_30 = "Sample log entry #40" in report and "Sample log entry #11" in report
                excludes_old = "Sample log entry #5\n" not in report
                # F-1225: a "clipboard copy" test that never reads the
                # clipboard proves nothing. Read it back and require the
                # pasted content to be the report that was generated.
                clipboard_text = app.clipboard().text()
                clipboard_ok = clipboard_text == report
                if not clipboard_ok:
                    return (
                        False,
                        f"Clipboard content != generated report "
                        f"(report={len(report)} chars, clipboard={len(clipboard_text)} chars)",
                    )
                ok = has_header and has_tail_30 and excludes_old
                return ok, f"Report generated & clipboard read-back verified ({len(report)} chars, last 30 log lines verified)"

        self._run_check(
            "SUBSYSTEM_SUPPORT",
            "SupportDiagnostics report & clipboard copy",
            _test_support_diagnostics,
        )

        # 4.6 BotWorker state machine transitions check
        def _test_state_machine() -> Tuple[bool, str]:
            from core.state_machine import BotMode, BotState, BotWorker

            # test_mode=True is kept ONLY so construction does not touch
            # real input hardware; the assertions below check real state
            # transitions and recorded reasons, not just boolean flags
            # (F-1226: the old test passed by reading is_paused alone).
            worker = BotWorker(config={"gathering": {"test_mode": True}})
            worker.set_mode(BotMode.MANUAL_RECORD.value)
            mode_manual = worker.mode == BotMode.MANUAL_RECORD.value
            worker.set_mode(BotMode.AUTO_RECORD.value)
            mode_auto = worker.mode == BotMode.AUTO_RECORD.value
            worker.set_mode(BotMode.ROUTE_MODE.value)
            # An unsupported mode must fall back to ROUTE_MODE (documented
            # set_mode behaviour) instead of silently keeping the old mode.
            worker.set_mode("bogus_mode_xyz")
            mode_fallback = worker.mode == BotMode.ROUTE_MODE.value
            worker.set_mode(BotMode.ROUTE_MODE.value)
            worker.pause("Unit test pause")
            paused_ok = (
                worker.is_paused
                and worker.state == BotState.PAUSED.value
                and worker.pause_reason == "Unit test pause"
            )
            worker.resume()
            resumed_ok = (
                not worker.is_paused
                and worker.state != BotState.PAUSED.value
                and worker.pause_reason == ""
            )
            ok = mode_manual and mode_auto and mode_fallback and paused_ok and resumed_ok
            return ok, f"BotWorker modes & pause/resume verified (state={worker.state}, mode={worker.mode})"

        self._run_check("SUBSYSTEM_CORE", "BotWorker state machine & mode transitions", _test_state_machine)

        # 4.7 UI Components instantiation check (OverlayHUD, F9Menu)
        def _test_ui_components() -> Tuple[bool, str]:
            app = self._ensure_qt_app()
            if app is None:
                return False, "QApplication unavailable"

            from ui.f9_menu import F9Menu
            from ui.overlay_hud import OverlayHUD

            hud = OverlayHUD()
            hud.update_state(
                {
                    "state": "GATHERING",
                    "mode": "route",
                    "title": "Gathering Orichalcum Ore",
                    "subtitle": "Prompt recognized at 94%",
                    "gathered": 5,
                    "f_pressed": 12,
                    "match_pct": 94,
                }
            )
            hud.add_log("Self-test log line")
            # F-1158: constructing the widgets and returning True proves
            # nothing. Read the applied state back off the widgets.
            badge_text = hud.badge_label.text() if hasattr(hud, "badge_label") else ""
            title_text = hud.title_label.text() if hasattr(hud, "title_label") else ""
            stat1_text = hud.stat1_value.text() if hasattr(hud, "stat1_value") else ""
            stat2_text = hud.stat2_value.text() if hasattr(hud, "stat2_value") else ""
            stat3_text = hud.stat3_value.text() if hasattr(hud, "stat3_value") else ""
            log_texts = [
                getattr(hud, f"log_label_{i}").text()
                for i in (1, 2, 3)
                if hasattr(hud, f"log_label_{i}")
            ]
            hud_ok = (
                badge_text == "TOPLANIYOR"
                and "Orichalcum" in title_text
                and stat1_text == "5"
                and stat2_text == "12"
                and "94" in stat3_text
                and any("Self-test log line" in t for t in log_texts)
            )
            if not hud_ok:
                return (
                    False,
                    f"OverlayHUD did not reflect update_state: badge={badge_text!r}, "
                    f"title={title_text!r}, stat1={stat1_text!r}, stat2={stat2_text!r}, "
                    f"stat3={stat3_text!r}, logs={log_texts!r}",
                )
            f9 = F9Menu(active_mode="route")
            f9.set_active_route("Altgard", "Dredgion Crash Site")
            f9_ok = (
                getattr(f9, "_active_region", "") == "Altgard"
                and getattr(f9, "_active_route", "") == "Dredgion Crash Site"
                and "Altgard" in f9._format_active_route_label()
                and "Dredgion Crash Site" in f9._format_active_route_label()
            )
            if not f9_ok:
                return (
                    False,
                    f"F9Menu did not reflect set_active_route: region={getattr(f9, '_active_region', '?')!r}, "
                    f"route={getattr(f9, '_active_route', '?')!r}",
                )
            return True, f"OverlayHUD({hud.width()}x{hud.height()}, badge={badge_text}, gathered={stat1_text}) & F9Menu(Altgard/Dredgion Crash Site) state verified by read-back"

        self._run_check("SUBSYSTEM_UI", "OverlayHUD & F9Menu headless verification", _test_ui_components)

    # =========================================================================
    # Phase 8: Real-Game Screenshot Audit, 3x Harvest Lifecycle & Minimap Auto-Recorder Tests
    # =========================================================================

    def run_phase8_real_screenshot_and_harvest_tests(self) -> None:
        """
        Phase 8: Real-Game Screenshot Audit (`pilgrims_ridge_1.jpg`, `minimap_filter_closeup.jpg`,
        `minimap_sprout_closeup.jpg`, `aria_3d_crest_far.jpg`, `aria_f_gather_not_on_minimap.jpg`,
        `mela_f_gather_not_on_minimap.jpg`), 3x Harvest Lifecycle & Minimap Auto-Recorder Tests.
        """
        root_str = str(self.root)
        ref_dir = os.path.join(root_str, "reference_screenshots")
        templates_dir = os.path.join(root_str, "templates")

        required_screenshots = (
            "pilgrims_ridge_1.jpg",
            "minimap_filter_closeup.jpg",
            "minimap_sprout_closeup.jpg",
            "aria_3d_crest_far.jpg",
            "aria_f_gather_not_on_minimap.jpg",
            "mela_f_gather_not_on_minimap.jpg",
        )
        required_phase8_templates = (
            "e_gather_icon.png",
            "minimap_filter_funnel.png",
            "minimap_feather_odyle.png",
            "minimap_blue_wing.png",
            "minimap_green_stack.png",
            "minimap_monster_dot.png",
            "minimap_gatherable_sprout_icon.png",
            "minimap_sprout_node.png",
            "world_gather_crest.png",
            "f_gather_aria.png",
        )

        # ---------------------------------------------------------------------
        # Test 8.1: Reference Screenshots & Extracted Templates Verification
        # ---------------------------------------------------------------------
        def _test_8_1_screenshots_and_templates() -> Tuple[bool, str]:
            loaded_refs: Dict[str, Tuple[int, int]] = {}
            missing_refs: List[str] = []
            for fname in required_screenshots:
                fpath = os.path.join(ref_dir, fname)
                if not os.path.isfile(fpath):
                    missing_refs.append(fname)
                    continue
                img = self._imread_cached(Path(fpath))
                if img is None or img.size == 0:
                    missing_refs.append(f"{fname}(unreadable)")
                    continue
                h, w = img.shape[:2]
                loaded_refs[fname] = (w, h)

            if missing_refs:
                return False, f"Missing/unreadable reference screenshots: {missing_refs}"

            # Ensure all templates are extracted if any are absent
            if any(not os.path.isfile(os.path.join(templates_dir, t)) for t in required_phase8_templates):
                from tools.template_extractor import extract_all_templates

                # F-1159: extract_all_templates only accepts
                # `project_root` (+ **kwargs it silently swallows); the
                # old call passed analysis_dir/output_dir/assets_dir/
                # reference_dir kwargs that were ignored, and ignored the
                # return value too. Call it with its real signature and
                # require the returned mapping to actually contain the
                # templates we need.
                extracted = extract_all_templates(project_root=self.root)
                if not isinstance(extracted, dict) or not extracted:
                    return False, "extract_all_templates returned nothing while templates were missing"
                extracted_names = {os.path.basename(str(v)) for v in extracted.values()}
                extracted_names |= {str(k) for k in extracted.keys()}
                still_missing = [
                    t for t in required_phase8_templates
                    if t not in extracted_names and not os.path.isfile(os.path.join(templates_dir, t))
                ]
                if still_missing:
                    return False, f"extract_all_templates did not produce: {still_missing}"

            missing_tmpls: List[str] = []
            for tname in required_phase8_templates:
                tpath = os.path.join(templates_dir, tname)
                if not os.path.isfile(tpath):
                    missing_tmpls.append(tname)
                    continue
                timg = self._imread_cached(Path(tpath))
                if timg is None or timg.size == 0:
                    missing_tmpls.append(f"{tname}(corrupt)")

            if missing_tmpls:
                return False, f"Missing/corrupt templates in templates/: {missing_tmpls}"

            pr_dim = loaded_refs["pilgrims_ridge_1.jpg"]
            mf_dim = loaded_refs["minimap_filter_closeup.jpg"]
            return (
                True,
                f"6 screenshots loaded (pilgrims={pr_dim[0]}x{pr_dim[1]}, filter={mf_dim[0]}x{mf_dim[1]}) & {len(required_phase8_templates)} templates verified",
            )

        self._run_check(
            "PHASE8_8.1",
            "Real screenshots & templates/ verification",
            _test_8_1_screenshots_and_templates,
        )

        # ---------------------------------------------------------------------
        # Test 8.2: RouteManager 3x Harvest Lifecycle & Minimap Auto-Recorder
        # ---------------------------------------------------------------------
        def _test_8_2_harvest_3x_lifecycle() -> Tuple[bool, str]:
            from core.auto_cartographer import AutoCartographer
            from core.route_manager import RouteManager

            pilgrims_path = os.path.join(ref_dir, "pilgrims_ridge_1.jpg")
            pilgrims_img = self._imread_cached(Path(pilgrims_path))
            if pilgrims_img is None:
                return False, "pilgrims_ridge_1.jpg could not be read"

            with tempfile.TemporaryDirectory() as tmp_dir:
                rm = RouteManager(base_dir=tmp_dir, default_rest_sec=300)
                rm.create_region("Pilgrims_Ridge")
                rm.create_route("Pilgrims_Ridge", "Odyle_Run")
                rm.active_region = "Pilgrims_Ridge"
                rm.active_route = "Odyle_Run"

                # Verify F8 instant minimap recording on real screenshot.
                # F-1276: asserting max_harvests == 3 on the DEFAULT proves
                # nothing (the default IS 3), and resource_type is just
                # the label the caller passed in. Pass max_harvests
                # explicitly, and prove the parameter (not the default)
                # drives the node by recording a second node with a
                # different explicit value.
                recorded_node = rm.record_current_node(
                    pilgrims_img,
                    region="Pilgrims_Ridge",
                    route="Odyle_Run",
                    resource_type="Odyle",
                    max_harvests=3,
                )
                if recorded_node is None or int(getattr(recorded_node, "max_harvests", 0)) != 3:
                    return False, f"record_current_node failed or explicit max_harvests=3 not honoured: {recorded_node}"
                if str(getattr(recorded_node, "resource_type", "")) != "Odyle":
                    return False, f"record_current_node did not store the passed resource_type: {getattr(recorded_node, 'resource_type', None)!r}"
                probe_node = rm.record_current_node(
                    pilgrims_img,
                    region="Pilgrims_Ridge",
                    route="Odyle_Run",
                    resource_type="Odyle",
                    max_harvests=2,
                    allow_duplicate=True,
                )
                if probe_node is None or int(getattr(probe_node, "max_harvests", 0)) != 2:
                    return False, f"max_harvests parameter ignored (expected 2, got {getattr(probe_node, 'max_harvests', None)})"

                nid = int(recorded_node.id)
                if not hasattr(rm, "record_harvest_stage"):
                    return False, "RouteManager missing record_harvest_stage method"

                # Stage 1/3 -> (1, 3, False) and node stays 'active'
                s1 = rm.record_harvest_stage(nid, rest_minutes=5.0, region="Pilgrims_Ridge")
                n1 = rm.get_node(nid, region="Pilgrims_Ridge")
                if tuple(s1) != (1, 3, False) or n1 is None or str(n1.state) != "active":
                    return False, f"Harvest 1/3 mismatch: res={s1}, state={getattr(n1, 'state', None)}"

                # Stage 2/3 -> (2, 3, False) and node stays 'active'
                s2 = rm.record_harvest_stage(nid, rest_minutes=5.0, region="Pilgrims_Ridge")
                n2 = rm.get_node(nid, region="Pilgrims_Ridge")
                if tuple(s2) != (2, 3, False) or n2 is None or str(n2.state) != "active":
                    return False, f"Harvest 2/3 mismatch: res={s2}, state={getattr(n2, 'state', None)}"

                # Stage 3/3 -> (3, 3, True) and node transitions to 'resting'
                s3 = rm.record_harvest_stage(nid, rest_minutes=5.0, region="Pilgrims_Ridge")
                n3 = rm.get_node(nid, region="Pilgrims_Ridge")
                if tuple(s3) != (3, 3, True) or n3 is None or str(n3.state) != "resting":
                    return False, f"Harvest 3/3 mismatch: res={s3}, state={getattr(n3, 'state', None)}"

                # Verify AutoCartographer minimap auto-recorder on sprout screenshot (3-leaf sprout only, no feathers)
                sprout_img = self._imread_cached(Path(os.path.join(ref_dir, "minimap_sprout_closeup.jpg")))
                carto = AutoCartographer(route_manager=rm)
                auto_mm_nodes = carto.record_from_minimap(
                    sprout_img,
                    route_manager=rm,
                    region="Pilgrims_Ridge",
                    route="Odyle_Run",
                )
                auto_3d_nodes = carto.record_from_3d_world(
                    pilgrims_img,
                    route_manager=rm,
                    region="Pilgrims_Ridge",
                    route="Odyle_Run",
                )
                if not auto_mm_nodes or any(int(x.get("max_harvests", 0)) < 3 for x in auto_mm_nodes):
                    return False, f"AutoCartographer.record_from_minimap failed on sprouts: {auto_mm_nodes}"
                if not auto_3d_nodes or any(int(x.get("max_harvests", 0)) < 3 for x in auto_3d_nodes):
                    return False, f"AutoCartographer.record_from_3d_world failed on pilgrims_ridge_1.jpg: {auto_3d_nodes}"

                return (
                    True,
                    f"1/3->{s1}(active), 2/3->{s2}(active), 3/3->{s3}(resting), auto_mm={len(auto_mm_nodes)}, auto_3d={len(auto_3d_nodes)}",
                )

        self._run_check(
            "PHASE8_8.2",
            "RouteManager 3x harvest lifecycle & minimap auto-recorder",
            _test_8_2_harvest_3x_lifecycle,
        )

        # ---------------------------------------------------------------------
        # Test 8.3: Minimap Sprout Detection vs Feather & Monster Dot Rejection
        # ---------------------------------------------------------------------
        def _test_8_3_minimap_scanner_real_screenshots() -> Tuple[bool, str]:
            from core.auto_cartographer import AutoCartographer
            from core.minimap_scanner import MinimapScanner

            scanner = MinimapScanner()
            carto = AutoCartographer()

            pr_img = self._imread_cached(Path(os.path.join(ref_dir, "pilgrims_ridge_1.jpg")))
            mf_img = self._imread_cached(Path(os.path.join(ref_dir, "minimap_filter_closeup.jpg")))
            sp_img = self._imread_cached(Path(os.path.join(ref_dir, "minimap_sprout_closeup.jpg")))
            aria_far_img = self._imread_cached(Path(os.path.join(ref_dir, "aria_3d_crest_far.jpg")))
            mela_f_img = self._imread_cached(Path(os.path.join(ref_dir, "mela_f_gather_not_on_minimap.jpg")))
            if any(x is None for x in (pr_img, mf_img, sp_img, aria_far_img, mela_f_img)):
                return False, "Failed to load minimap test screenshots"

            # Verify white feathers AND plain circular white Monster dots (•) are REJECTED on minimap,
            # and ONLY the white 3-leaf sprout icon ('Gatherable Material') is accepted on minimap!
            pr_filtered = carto._detect_minimap_gatherables(pr_img)
            mf_filtered = carto._detect_minimap_gatherables(mf_img)
            sp_filtered = carto._detect_minimap_gatherables(sp_img)
            aria_mm_filtered = carto._detect_minimap_gatherables(aria_far_img)
            mela_mm_filtered = carto._detect_minimap_gatherables(mela_f_img)

            # pilgrims_ridge_1.jpg, minimap_filter_closeup.jpg, and mela_f_gather_not_on_minimap.jpg
            # have feathers and monster dots on the minimap, NO 3-leaf sprouts -> must be 0
            if len(pr_filtered) != 0:
                return False, f"Feather or monster dot false-positived on pilgrims_ridge_1.jpg minimap: {pr_filtered}"
            if len(mf_filtered) != 0:
                return False, f"Feather or monster dot false-positived on minimap_filter_closeup.jpg: {mf_filtered}"
            if len(mela_mm_filtered) != 0:
                return False, f"Feather or monster dot false-positived on mela_f_gather_not_on_minimap.jpg: {mela_mm_filtered}"

            # minimap_sprout_closeup.jpg has 2 white 3-leaf sprout icons and 2 round white monster dots -> must detect 2 sprouts
            if len(sp_filtered) != 2 or any(d.get("resource_type") != "Gatherable Material" for d in sp_filtered):
                return (
                    False,
                    f"minimap_sprout_closeup.jpg expected 2 'Gatherable Material' sprouts (rejecting 2 monster dots), got: {sp_filtered}",
                )

            # aria_3d_crest_far.jpg has a white 3-leaf sprout icon on the minimap -> must detect >= 1 'Gatherable Material'
            if len(aria_mm_filtered) < 1 or any(d.get("resource_type") != "Gatherable Material" for d in aria_mm_filtered):
                return (
                    False,
                    f"aria_3d_crest_far.jpg expected >= 1 'Gatherable Material' sprout on minimap, got: {aria_mm_filtered}",
                )

            # Also verify MinimapScanner directly rejects feathers & monster dots on pilgrims_ridge_1.jpg & minimap_filter_closeup.jpg
            # while detecting 3-leaf sprout resources on minimap_sprout_closeup.jpg and aria_3d_crest_far.jpg
            ms_pr = scanner.scan_minimap_resources_and_odyle(pr_img)
            ms_mf = scanner.scan_minimap_resources_and_odyle(mf_img)
            ms_sp = scanner.scan_minimap_resources_and_odyle(sp_img)
            ms_aria = scanner.scan_minimap_resources_and_odyle(aria_far_img)
            if len(ms_pr) != 0 or len(ms_mf) != 0:
                return False, f"MinimapScanner failed to reject feathers/monster dots: pr={ms_pr}, mf={ms_mf}"
            if len(ms_sp) < 1 or len(ms_aria) < 1:
                return False, f"MinimapScanner missed 3-leaf sprout on sprout closeup or aria_3d_crest_far: sp={ms_sp}, aria={ms_aria}"
            hostile_fp = [d for d in (ms_sp + ms_aria) if d.get("is_hostile", False)]
            if hostile_fp:
                return False, f"MinimapScanner misclassified white monster dots as red hostile dots: {hostile_fp}"

            return (
                True,
                f"sprouts_closeup={len(sp_filtered)}(ms={len(ms_sp)}), aria_mm={len(aria_mm_filtered)}(ms={len(ms_aria)}), feathers&monsters_rejected=(pr=0, mf=0, mela=0)",
            )

        self._run_check(
            "PHASE8_8.3",
            "MinimapScanner gatherable vs Monster dot filtering",
            _test_8_3_minimap_scanner_real_screenshots,
        )

        # ---------------------------------------------------------------------
        # Test 8.4: MaterialScanner & VisionEngine on Pilgrim's Ridge, Aria & Mela Screenshots
        # ---------------------------------------------------------------------
        def _test_8_4_material_scanner_and_vision_real_screenshots() -> Tuple[bool, str]:
            from core.auto_cartographer import AutoCartographer
            from core.material_scanner import MaterialScanner
            from core.vision import VisionEngine

            mat_scanner = MaterialScanner()
            vision = VisionEngine()
            carto = AutoCartographer()

            pr_img = self._imread_cached(Path(os.path.join(ref_dir, "pilgrims_ridge_1.jpg")))
            aria_far_img = self._imread_cached(Path(os.path.join(ref_dir, "aria_3d_crest_far.jpg")))
            aria_f_img = self._imread_cached(Path(os.path.join(ref_dir, "aria_f_gather_not_on_minimap.jpg")))
            mela_f_img = self._imread_cached(Path(os.path.join(ref_dir, "mela_f_gather_not_on_minimap.jpg")))

            if any(x is None for x in (pr_img, aria_far_img, aria_f_img, mela_f_img)):
                return False, "Could not load Pilgrim's Ridge / Aria / Mela screenshots"

            # 1. Pilgrim's Ridge: Must NOT false-positive on 'Yellow-Maned Zaif' monster nameplate
            # and MUST detect floating [E] gatherable plant icon and/or minimap materials
            pr_3d_nameplates = mat_scanner.scan_3d_world_nameplates(pr_img)
            pr_all_mats = mat_scanner.scan_frame_for_materials(pr_img)
            pr_e_icons = carto._detect_3d_world_gatherables(pr_img)

            zaif_fp = [
                m
                for m in pr_3d_nameplates
                if "zaif" in str(getattr(m, "material_type", "")).lower()
                or (450 <= int(getattr(m, "screen_x", 0)) <= 750 and 150 <= int(getattr(m, "screen_y", 0)) <= 360)
            ]
            if zaif_fp:
                return False, f"False positive on Yellow-Maned Zaif monster nameplate: {zaif_fp}"
            # F-1218: the old `if not pr_all_mats and not pr_e_icons`
            # passed when EITHER subsystem found something, hiding a
            # collapse of the other. BOTH must produce results.
            if not pr_all_mats or not pr_e_icons:
                return (
                    False,
                    f"Pilgrim's Ridge subsystem collapse: MaterialScanner found "
                    f"{len(pr_all_mats)} materials, Cartographer 3D found "
                    f"{len(pr_e_icons)} [E] icons — both must be non-empty",
                )

            # 2. VisionEngine.detect_gather_prompt on aria_f_gather_not_on_minimap.jpg and mela_f_gather_not_on_minimap.jpg
            # F-1219: NEVER read last_detected_gather_key with a default
            # that equals the expected value ("f"/"e"). A missing
            # attribute used to pass as the right key. Require the
            # attribute to exist and read it without a default.
            if not hasattr(vision, "last_detected_gather_key"):
                return False, "VisionEngine exposes no last_detected_gather_key attribute"
            aria_f_found, aria_f_conf, aria_f_bbox = vision.detect_gather_prompt(aria_f_img)
            aria_key = vision.last_detected_gather_key
            mela_f_found, mela_f_conf, mela_f_bbox = vision.detect_gather_prompt(mela_f_img)
            mela_key = vision.last_detected_gather_key
            pr_found, _, _ = vision.detect_gather_prompt(pr_img)
            pr_key = vision.last_detected_gather_key

            if not aria_f_found or aria_f_bbox is None or aria_key != "f":
                return False, f"VisionEngine.detect_gather_prompt missed 'F Gather' on aria_f_gather_not_on_minimap.jpg (conf={aria_f_conf}, key={aria_key})"
            if not mela_f_found or mela_f_bbox is None or mela_key != "f":
                return False, f"VisionEngine.detect_gather_prompt missed 'F Gather' on mela_f_gather_not_on_minimap.jpg (conf={mela_f_conf}, key={mela_key})"
            if pr_found and pr_key == "f":
                return False, "VisionEngine.detect_gather_prompt false-positived 'F Gather' on pilgrims_ridge_1.jpg"

            # 3. Verify 3D world detection of Aria & Mela even when not shown on minimap
            aria_far_3d = carto._detect_3d_world_gatherables(aria_far_img)
            aria_f_3d = carto._detect_3d_world_gatherables(aria_f_img)
            mela_f_3d = carto._detect_3d_world_gatherables(mela_f_img)

            aria_f_mats = mat_scanner.scan_frame_for_materials(aria_f_img)
            mela_f_mats = mat_scanner.scan_frame_for_materials(mela_f_img)

            aria_f_prompts = [m for m in aria_f_mats if getattr(m, "source", "") in ("gather_prompt", "3d_world")]
            mela_f_prompts = [m for m in mela_f_mats if getattr(m, "source", "") in ("gather_prompt", "3d_world")]

            if not aria_far_3d or not any(d.get("resource_type") == "Aria" for d in aria_far_3d):
                return False, f"3D world Aria crest missed on aria_3d_crest_far.jpg: {aria_far_3d}"
            if (not aria_f_3d or not any(d.get("resource_type") == "Aria" for d in aria_f_3d)) or not aria_f_prompts:
                return False, f"Aria 3D/prompt missed on aria_f_gather_not_on_minimap.jpg: 3d={aria_f_3d}, mats={len(aria_f_prompts)}"
            if (not mela_f_3d or not any(d.get("resource_type") == "Mela" for d in mela_f_3d)) or not mela_f_prompts:
                return False, f"Mela 3D/prompt missed on mela_f_gather_not_on_minimap.jpg: 3d={mela_f_3d}, mats={len(mela_f_prompts)}"

            return (
                True,
                f"Zaif FP=0, pilgrims_mats={len(pr_all_mats)}([E]={len(pr_e_icons)}), Aria_F={aria_f_conf}%, Mela_F={mela_f_conf}%, 3D(Aria/Mela)=OK",
            )

        self._run_check(
            "PHASE8_8.4",
            "MaterialScanner & VisionEngine on Pilgrim's Ridge, Aria & Mela",
            _test_8_4_material_scanner_and_vision_real_screenshots,
        )

        # ---------------------------------------------------------------------
        # Test 8.5: GameHUDReader Chat Alerts & Monster Target Bar vs Cast Bar
        # ---------------------------------------------------------------------
        def _test_8_5_game_hud_reader() -> Tuple[bool, str]:
            from core.ocr_distance import GameHUDReader

            pr_img = self._imread_cached(Path(os.path.join(ref_dir, "pilgrims_ridge_1.jpg")))
            if pr_img is None:
                return False, "Failed to load pilgrims_ridge_1.jpg"

            reader = GameHUDReader()
            alerts = reader.detect_chat_system_alerts(pr_img)
            target_info = reader.detect_top_target_bar(pr_img)
            cast_progress = reader.read_cast_bar_progress(pr_img)
            is_gathering, gather_prog = reader.verify_gather_cast_bar(pr_img)

            if not bool(alerts.get("extraction_failed")) or not bool(alerts.get("cannot_mount")):
                return False, f"Chat alerts missed on pilgrims_ridge_1.jpg: {alerts}"

            if (
                not bool(target_info.get("has_target"))
                or not bool(target_info.get("is_monster_target"))
                or bool(target_info.get("is_gather_target"))
            ):
                return False, f"Top target bar failed to classify Yellow-Maned Zaif as monster: {target_info}"

            if float(cast_progress) > 0.05 or is_gathering:
                return (
                    False,
                    f"Monster target bar confused with gather cast bar: progress={cast_progress}, verify=({is_gathering}, {gather_prog})",
                )

            return (
                True,
                f"alerts=(extract_fail={alerts['extraction_failed']}, no_mount={alerts['cannot_mount']}), monster_target={target_info['is_monster_target']}, cast={cast_progress:.2f}",
            )

        self._run_check(
            "PHASE8_8.5",
            "GameHUDReader chat alerts & monster target bar distinction",
            _test_8_5_game_hud_reader,
        )

        # ---------------------------------------------------------------------
        # Test 8.6: CombatRecovery Player HP & Alive Verification on Pilgrim's Ridge
        # ---------------------------------------------------------------------
        def _test_8_6_combat_recovery_pilgrims_ridge() -> Tuple[bool, str]:
            from core.combat_recovery import CombatRecovery

            pr_img = self._imread_cached(Path(os.path.join(ref_dir, "pilgrims_ridge_1.jpg")))
            if pr_img is None:
                return False, "Failed to load pilgrims_ridge_1.jpg"

            cr = CombatRecovery()
            hp_ratio = float(cr.get_player_hp_ratio(pr_img))
            is_dead = bool(cr.check_character_dead(pr_img))

            if hp_ratio <= 0.85:
                return False, f"Expected full player HP (> 0.85) on pilgrims_ridge_1.jpg, got {hp_ratio:.3f}"
            if is_dead:
                return False, "check_character_dead returned True on alive player in pilgrims_ridge_1.jpg"

            return True, f"pilgrims_ridge_1.jpg player HP={hp_ratio:.3f} (> 0.85), check_character_dead={is_dead}"

        self._run_check(
            "PHASE8_8.6",
            "CombatRecovery full HP (> 0.85) & alive check on Pilgrim's Ridge",
            _test_8_6_combat_recovery_pilgrims_ridge,
        )

        # ---------------------------------------------------------------------
        # Test 8.7: Full AST Zero-Placeholder Gate Across All Repository .py Files
        # ---------------------------------------------------------------------
        def _test_8_7_ast_zero_placeholder_gate() -> Tuple[bool, str]:
            # D72: the scan itself lives in _scan_repo_placeholders(),
            # which reuses the dynamic phase's memoized per-file results
            # (same _analyze_ast pass) and only parses the files the
            # dynamic phase skips. Verdict rules are the merged ones:
            # the S-4702 literal-claim tightening lives in _analyze_ast.
            violations, n_files = self._scan_repo_placeholders()

            if violations:
                return False, f"AST Placeholder violations ({len(violations)}): {', '.join(violations[:5])}"

            return True, f"Zero placeholders across {n_files} Python files (AST & comment audit clean)"

        self._run_check(
            "PHASE8_8.7",
            "Full AST Zero-Placeholder Gate across repository",
            _test_8_7_ast_zero_placeholder_gate,
        )

    # =========================================================================
    # Table Formatter & Runner
    # =========================================================================

    def format_report_table(self) -> str:
        """Formats all test results into a clean, aligned ASCII verification table."""
        # F-1273: write PASS/FAIL plainly. The old "P" + "ASS" split was
        # presumably to dodge a naive text scan; a verification report
        # must say PASS/FAIL in plain text.
        status_pass = "PASS"
        status_fail = "FAIL"

        headers = ("#", "CATEGORY", "TEST CASE", "TIME (ms)", "STATUS", "DETAILS")
        rows: List[Tuple[str, str, str, str, str, str]] = []

        for idx, r in enumerate(self.results, start=1):
            badge = status_pass if r.passed else status_fail
            # F-1274: do not truncate details (the old r.details[:68] hid
            # the very evidence a failure report exists to show).
            rows.append(
                (
                    f"{idx:02d}",
                    r.category,
                    r.name,
                    f"{r.duration_ms:7.2f}",
                    badge,
                    r.details,
                )
            )

        col_widths = [len(h) for h in headers]
        for row in rows:
            for c_idx, cell in enumerate(row):
                if len(cell) > col_widths[c_idx]:
                    col_widths[c_idx] = len(cell)

        sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        header_line = (
            "| "
            + " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
            + " |"
        )

        lines: List[str] = [
            "",
            "=" * len(sep),
            "GATHERION A2 BOT - FULL COMPREHENSIVE SELF-TEST SUITE REPORT",
            "=" * len(sep),
            sep,
            header_line,
            sep,
        ]

        for row in rows:
            line = (
                "| "
                + " | ".join(row[i].ljust(col_widths[i]) for i in range(len(row)))
                + " |"
            )
            lines.append(line)

        passed_cnt = sum(1 for r in self.results if r.passed)
        failed_cnt = len(self.results) - passed_cnt
        total_ms = sum(r.duration_ms for r in self.results)
        overall = status_pass if failed_cnt == 0 else status_fail

        lines.append(sep)
        lines.append(
            f"SUMMARY: Total={len(self.results)} | {status_pass}={passed_cnt} | "
            f"{status_fail}={failed_cnt} | Duration={total_ms:.2f} ms | OVERALL={overall}"
        )
        lines.append("=" * len(sep))
        return "\n".join(lines)

    def run_all(self, categories: Optional[List[str]] = None) -> bool:
        """Executes the verification suite and prints the summary table.

        D72: `categories` runs only those check categories (same as
        `--only` on the CLI), so iterating on one failing check no longer
        costs a full ~4-minute run. The filter is explicit: a banner
        names the selection, and the SUMMARY Total counts only the checks
        that actually ran. With no filter the behaviour is unchanged.
        """
        self.results.clear()
        if categories:
            unknown = sorted(set(categories) - CHECK_CATEGORIES)
            if unknown:
                raise ValueError(
                    f"Unknown check categories: {unknown}. "
                    f"Valid categories: {sorted(CHECK_CATEGORIES)}"
                )
            self._only_categories = frozenset(categories)
            print(
                "[self_test_suite] FILTER active: only categories "
                f"{sorted(self._only_categories)} will run "
                f"({len(self._only_categories)} of {len(CHECK_CATEGORIES)}); "
                "all other checks are NOT run in this invocation."
            )
        try:
            self.run_dynamic_module_checks()
            self.run_asset_and_template_checks()
            self.run_video_frame_vision_tests()
            self.run_functional_subsystem_tests()
            self.run_phase8_real_screenshot_and_harvest_tests()
        finally:
            self._only_categories = None

        table_str = self.format_report_table()
        print(table_str)
        return all(r.passed for r in self.results)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: full run, `--only` subset, `--list`, `--selftest`."""
    parser = argparse.ArgumentParser(
        description="Gatherion A2 Bot self-test suite.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="CATEGORY",
        help="Run only these check categories (see --list).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the check categories and exit.",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run the suite's built-in unit tests for its own helpers and exit.",
    )
    args = parser.parse_args(argv)
    if args.list:
        for category in sorted(CHECK_CATEGORIES):
            print(category)
        return 0
    if args.selftest:
        return _run_builtin_selftests()
    suite = SelfTestSuite(PROJECT_ROOT)
    all_ok = suite.run_all(categories=args.only)
    return 0 if all_ok else 1


def _run_builtin_selftests() -> int:
    """Built-in unit tests for this file's own helpers (D72).

    Run: ``python tools/self_test_suite.py --selftest``. Covers only
    the D72 changes: the shared AST analysis + memoization, the image
    cache, the --only filter, CHECK_CATEGORIES coverage, and the
    template early-exit verdict equivalence. Uses synthetic roots under
    ~/workspace (never /tmp) so the real repo is untouched. Returns 0
    when every check passes, 1 otherwise.
    """
    failures: List[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"[selftest] {'PASS' if cond else 'FAIL'} {name}"
              + (f" :: {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    # --- 1. _analyze_ast: flags stubs, ignores real code --------------
    stub_src = (
        '"""Module docstring."""\n'
        "\n"
        "def stub_pass():\n"
        '    """Doc."""\n'
        "    pass\n"
        "\n"
        "def stub_ellipsis():\n"
        "    ...\n"
        "\n"
        "def get_thing_ready_flag():\n"
        "    return True\n"
        "\n"
        "def get_widget_readiness_status():\n"
        '    return {"ready": True, "passed": 1}\n'
        "\n"
        # NOTE: the marker is built from two literals so this file
        # itself contains no placeholder-marker comment line for
        # the 8.7 gate to flag; at runtime the analyzed source
        # still carries a real marker line, which is what the
        # rule must catch.
        "# TO" "DO: finish this later\n"
        "def real_function(x):\n"
        "    total = x + 1\n"
        "    return total\n"
        "\n"
        "def get_count():\n"
        "    return 42\n"
    )
    node_count, violations = _analyze_ast("synthetic.py", stub_src, ast.parse(stub_src))
    check("ast: node count positive", node_count > 30, f"nodes={node_count}")
    for needle in ("stub_pass", "stub_ellipsis", "get_thing_ready_flag",
                   "get_widget_readiness_status", "placeholder comment"):
        check(f"ast: flags {needle}", any(needle in v for v in violations),
              f"violations={violations}")
    for needle in ("real_function", "get_count"):
        check(f"ast: ignores {needle}", not any(needle in v for v in violations),
              f"violations={violations}")
    check("ast: exactly 5 violations", len(violations) == 5, f"got {len(violations)}: {violations}")

    # --- 2. Memoization: gate reuses the dynamic phase's parse --------
    with tempfile.TemporaryDirectory(dir=str(Path.home() / "workspace")) as td:
        troot = Path(td)
        for pkg in ("config", "core", "ui", "tools"):
            (troot / pkg).mkdir()
            (troot / pkg / "__init__.py").write_text(
                "def get_pkg_ready_flag():\n    return True\n", encoding="utf-8")
        (troot / "config" / "stub_mod.py").write_text(
            "def stub_pass():\n    pass\n# TO" "DO: later\n", encoding="utf-8")
        (troot / "core" / "clean_mod.py").write_text(
            "def compute(x):\n    return x * 2\n", encoding="utf-8")
        suite = SelfTestSuite(root=troot)
        parse_files: List[str] = []
        real_parse = ast.parse
        # The dynamic phase parses with absolute filenames, the gate
        # with repo-relative ones; match by relative suffix either way.
        fake_suffixes = (
            "config/stub_mod.py", "core/clean_mod.py",
            "config/__init__.py", "core/__init__.py",
            "ui/__init__.py", "tools/__init__.py",
        )

        def counting_parse(source, filename="<unknown>", *args, **kwargs):
            if any(str(filename).endswith(s) for s in fake_suffixes):
                parse_files.append(str(filename))
            return real_parse(source, filename=filename, *args, **kwargs)

        ast.parse = counting_parse  # type: ignore[assignment]
        try:
            suite.run_dynamic_module_checks()
            n_after_dynamic = len(parse_files)
            violations2, n_files = suite._scan_repo_placeholders()
            n_after_gate = len(parse_files)
        finally:
            ast.parse = real_parse  # type: ignore[assignment]
        # Dynamic phase parses the 2 real modules; the gate must only
        # parse the 4 __init__.py files it alone covers (2 + 4 = 6).
        check("memo: dynamic phase parsed 2 modules", n_after_dynamic == 2,
              f"parsed={n_after_dynamic}")
        check("memo: gate re-parsed only the 4 inits", n_after_gate == n_after_dynamic + 4,
              f"dynamic={n_after_dynamic} gate={n_after_gate}")
        check("memo: gate scanned 6 files", n_files == 6, f"n_files={n_files}")
        check("memo: gate found the stub + init stubs",
              any("stub_mod.py:stub_pass" in v for v in violations2)
              and sum(1 for v in violations2 if "get_pkg_ready_flag" in v) == 4,
              f"violations={violations2}")
        check("memo: gate found no clean_mod violation",
              not any("clean_mod" in v for v in violations2), f"violations={violations2}")

    # --- 3. Image cache: one decode per flag, copies isolate ---------
    suite2 = SelfTestSuite()
    frame = PROJECT_ROOT / "video_frames" / "frame_00s.jpg"
    decode_calls = {"n": 0}
    real_imdecode = cv2.imdecode

    def counting_imdecode(*args, **kwargs):
        decode_calls["n"] += 1
        return real_imdecode(*args, **kwargs)

    cv2.imdecode = counting_imdecode  # type: ignore[assignment]
    try:
        a = suite2._imread_cached(frame)
        b = suite2._imread_cached(frame)
        c = suite2._imread_cached(frame, gray=True)
        d = suite2._imread_cached(frame, gray=True)
    finally:
        cv2.imdecode = real_imdecode  # type: ignore[assignment]
    check("imgcache: 2 decodes for 4 reads (color+gray)",
          decode_calls["n"] == 2, f"decodes={decode_calls['n']}")
    check("imgcache: copies are equal but distinct objects",
          a is not None and b is not None and a is not b and bool(np.array_equal(a, b)))
    check("imgcache: gray copy has 2 dims", c is not None and d is not None and c.ndim == 2)
    if a is not None and b is not None:
        a[0, 0, 0] = (int(a[0, 0, 0]) + 1) % 256
        e = suite2._imread_cached(frame)
        check("imgcache: mutating a copy leaves the cache intact",
              e is not None and bool(np.array_equal(b, e)))

    # --- 4. --only filter: unselected checks never execute ------------
    suite3 = SelfTestSuite()
    suite3._only_categories = frozenset({"VIDEO_FRAMES"})
    try:
        suite3.run_asset_and_template_checks()
    finally:
        suite3._only_categories = None
    check("only: only VIDEO_FRAMES ran",
          len(suite3.results) == 5
          and all(r.category == "VIDEO_FRAMES" for r in suite3.results),
          f"results={[(r.category, r.name) for r in suite3.results]}")

    # --- 5. CHECK_CATEGORIES covers every _run_check call site --------
    own_source = Path(__file__).read_text(encoding="utf-8")
    found = set(re.findall(r'_run_check\(\s*"([A-Z0-9_.]+)"', own_source))
    check("categories: CHECK_CATEGORIES matches call sites",
          found == set(CHECK_CATEGORIES),
          f"missing={set(CHECK_CATEGORIES) - found} extra={found - set(CHECK_CATEGORIES)}")

    # --- 6. Template early exit: verdict identical, work skipped ------
    tmpl_names = (
        "f_gather_template.png", "f_gather.png", "f_gather_pill.png",
        "player_arrow.png", "map_header.png",
    )
    with tempfile.TemporaryDirectory(dir=str(Path.home() / "workspace")) as td:
        troot = Path(td)
        (troot / "templates").mkdir()
        (troot / "video_frames").mkdir()
        rng = np.random.default_rng(0)
        tmpls = {}
        for i, name in enumerate(tmpl_names):
            t = rng.integers(0, 256, size=(16, 16)).astype(np.uint8)
            tmpls[name] = t
            cv2.imwrite(str(troot / "templates" / name), t)
        # Frame A contains every template (match ~1.0); frame B is
        # unrelated noise (must stay < 0.75 for every template).
        frame_a = rng.integers(0, 256, size=(128, 128)).astype(np.uint8)
        for i, name in enumerate(tmpl_names):
            frame_a[8 + i * 20:24 + i * 20, 8:24] = tmpls[name]
        frame_b = rng.integers(0, 256, size=(128, 128)).astype(np.uint8)
        frame_c = rng.integers(0, 256, size=(128, 128)).astype(np.uint8)
        cv2.imwrite(str(troot / "video_frames" / "frame_00s.jpg"), frame_a)
        cv2.imwrite(str(troot / "video_frames" / "frame_01s.jpg"), frame_b)
        # Premise check (direct computation, not the code under test).
        premise_ok = True
        for name, t in tmpls.items():
            for neg in (frame_b, frame_c):
                s = float(cv2.matchTemplate(neg, t, cv2.TM_CCOEFF_NORMED).max())
                if s >= 0.75:
                    premise_ok = False
        check("tmpl: negative frames stay below 0.75 (premise)", premise_ok)
        match_calls = {"n": 0}
        real_match = cv2.matchTemplate

        def counting_match(*args, **kwargs):
            match_calls["n"] += 1
            return real_match(*args, **kwargs)

        cv2.matchTemplate = counting_match  # type: ignore[assignment]
        try:
            suite4 = SelfTestSuite(root=troot)
            ok_pass, details_pass = suite4._verify_templates()
        finally:
            cv2.matchTemplate = real_match  # type: ignore[assignment]
        check("tmpl: early exit passes with first qualifying ref", ok_pass,
              f"details={details_pass}")
        check("tmpl: early exit used 1 match per template (not 2)",
              match_calls["n"] == len(tmpl_names),
              f"matchTemplate calls={match_calls['n']}")
        check("tmpl: report says first >= 0.75", "[first >= 0.75]" in details_pass,
              f"details={details_pass}")

        # Failure path: only negative frames -> must FAIL and report the
        # TRUE best score across all references (no early exit to hide it).
        # The direct best is computed from the same JPEG-decoded bytes
        # the code under test sees, so the numbers must agree closely.
        (troot / "video_frames" / "frame_00s.jpg").unlink()
        cv2.imwrite(str(troot / "video_frames" / "frame_02s.jpg"), frame_c)
        neg_grays = []
        for fname in ("frame_01s.jpg", "frame_02s.jpg"):
            neg_bgr = safe_imread_color(troot / "video_frames" / fname)
            assert neg_bgr is not None
            neg_grays.append(cv2.cvtColor(neg_bgr, cv2.COLOR_BGR2GRAY))
        tmpl_gray = safe_imread_gray(troot / "templates" / "f_gather_template.png")
        assert tmpl_gray is not None
        direct_best = max(
            float(cv2.matchTemplate(neg, tmpl_gray, cv2.TM_CCOEFF_NORMED).max())
            for neg in neg_grays
        )
        suite5 = SelfTestSuite(root=troot)
        ok_fail, details_fail = suite5._verify_templates()
        m = re.search(r"best score=([0-9.]+)", details_fail)
        reported_best = float(m.group(1)) if m else -1.0
        check("tmpl: no-match root still fails", not ok_fail, f"details={details_fail}")
        check("tmpl: failure reports the true best score",
              m is not None and abs(reported_best - direct_best) < 0.005,
              f"reported={reported_best} direct={direct_best:.3f}")

    print(f"[selftest] {len(failures)} failure(s)" if failures else "[selftest] all checks passed")
    return 1 if failures else 0


def _run_real_self_test() -> dict:
    """Runs the full SelfTestSuite once per process and caches the outcome.

    F-1124: the old get_self_test_suite_* helpers returned
    ready/audit_passed/certified=True without running a single test.
    Every status helper below derives its answer from this real run.
    The suite's own run_* methods are called directly (no table print),
    and re-entrancy is guarded so a helper can never trigger a second,
    nested run.
    """
    cache = _run_real_self_test.__dict__.setdefault("cache", {})
    if "result" in cache:
        return cache["result"]
    if cache.get("running"):
        return {"total": 0, "passed": 0, "failed": 0, "all_passed": False,
                "failures": ["self-test run already in progress (re-entrant call)"]}
    cache["running"] = True
    try:
        suite = SelfTestSuite(PROJECT_ROOT)
        suite.run_dynamic_module_checks()
        suite.run_asset_and_template_checks()
        suite.run_video_frame_vision_tests()
        suite.run_functional_subsystem_tests()
        suite.run_phase8_real_screenshot_and_harvest_tests()
        passed = sum(1 for r in suite.results if r.passed)
        failures = [f"{r.category}:{r.name}" for r in suite.results if not r.passed]
        result = {
            "total": len(suite.results),
            "passed": passed,
            "failed": len(suite.results) - passed,
            "all_passed": bool(suite.results) and passed == len(suite.results),
            "failures": failures,
        }
    except Exception as exc:
        logger.error("Real self-test run failed: %s", exc)
        result = {"total": 0, "passed": 0, "failed": 0, "all_passed": False,
                  "failures": [f"self-test run raised: {exc}"]}
    finally:
        cache["running"] = False
    cache["result"] = result
    return result


def get_self_test_suite_summary() -> dict:
    """Returns the outcome of an actual SelfTestSuite run (never a canned True)."""
    real = _run_real_self_test()
    return {
        "self_test_ready": bool(real["all_passed"]),
        "passed": real["passed"],
        "failed": real["failed"],
        "total": real["total"],
        "failures": real["failures"],
    }


def get_self_test_suite_ready_flag() -> dict:
    """Returns the ready flag derived from an actual SelfTestSuite run."""
    real = _run_real_self_test()
    return {
        "module": "self_test_suite",
        "ready": bool(real["all_passed"]),
        "passed": real["passed"],
        "failed": real["failed"],
        "total": real["total"],
    }


if __name__ == "__main__":
    sys.exit(main())


def get_self_test_suite_version_info() -> dict:
    """Returns version info; `ready` reflects an actual SelfTestSuite run."""
    real = _run_real_self_test()
    return {"module": "self_test_suite", "version": __version__, "ready": bool(real["all_passed"])}


def get_self_test_suite_diagnostic_snapshot() -> dict:
    real = _run_real_self_test()
    return {
        "module": "self_test_suite",
        "status": "ok" if real["all_passed"] else "failed",
        "ready": bool(real["all_passed"]),
        "passed": real["passed"],
        "failed": real["failed"],
        "total": real["total"],
    }


def get_self_test_suite_capabilities() -> list:
    return ["self_test_suite", "diagnostic_snapshot", "version_info"]


def validate_self_test_suite_contract() -> bool:
    # F-1124: a contract is not valid because a capabilities list is
    # non-empty. It is valid only when the real suite run passes.
    return bool(get_self_test_suite_capabilities()) and bool(_run_real_self_test()["all_passed"])


def get_self_test_suite_baseline_metrics() -> dict:
    return {"module": "self_test_suite", "contract_valid": validate_self_test_suite_contract(), "capability_count": len(get_self_test_suite_capabilities())}


def get_self_test_suite_readiness_status() -> dict:
    try:
        return {
            "module": "self_test_suite",
            "ready": validate_self_test_suite_contract(),
            "metrics": get_self_test_suite_baseline_metrics(),
        }
    except Exception:
        return {"module": "self_test_suite", "ready": False, "metrics": {}}


def get_self_test_suite_audit_record() -> dict:
    try:
        return {
            "module": "self_test_suite",
            "audit_passed": validate_self_test_suite_contract(),
            "readiness": get_self_test_suite_readiness_status(),
        }
    except Exception:
        return {"module": "self_test_suite", "audit_passed": False, "readiness": {}}


def get_self_test_suite_operational_digest() -> dict:
    return {
        "module": "self_test_suite",
        "operational": validate_self_test_suite_contract(),
        "audit": get_self_test_suite_audit_record(),
    }


def get_self_test_suite_telemetry_payload() -> dict:
    return {
        "module": "self_test_suite",
        "telemetry_ready": validate_self_test_suite_contract(),
        "digest": get_self_test_suite_operational_digest(),
    }


def get_self_test_suite_integrity_seal() -> dict:
    return {
        "module": "self_test_suite",
        "sealed": validate_self_test_suite_contract(),
        "telemetry": get_self_test_suite_telemetry_payload(),
    }


def get_self_test_suite_compliance_certificate() -> dict:
    return {
        "module": "self_test_suite",
        "certified": validate_self_test_suite_contract(),
        "seal": get_self_test_suite_integrity_seal(),
    }


def get_self_test_suite_readiness_probe() -> dict:
    return {
        "module": "self_test_suite",
        "ready": validate_self_test_suite_contract(),
        "certificate": get_self_test_suite_compliance_certificate(),
    }


def get_self_test_suite_audit_summary() -> dict:
    return {
        "module": "self_test_suite",
        "audit_passed": validate_self_test_suite_contract(),
        "probe": get_self_test_suite_readiness_probe(),
    }


def get_self_test_suite_heartbeat_snapshot() -> dict:
    return {
        "module": "self_test_suite",
        "alive": validate_self_test_suite_contract(),
        "audit": get_self_test_suite_audit_summary(),
    }



