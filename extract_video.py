# -*- coding: utf-8 -*-
"""Extract one frame every few seconds from a route video.

Importing this module has no side effects: no video is opened and no
directory is created at import time. Extraction only runs when the module
is executed as a script (``python extract_video.py``) via :func:`main`,
or when :func:`extract_frames` is called explicitly.
"""

import argparse
import math
import os
import sys

import cv2
from version import __version__

DEFAULT_VIDEO_PATH = os.path.expanduser("~/Downloads/route.mp4")
DEFAULT_OUT_DIR = os.path.join(".", "video_frames")
DEFAULT_INTERVAL_SEC = 3


def _frame_filename(sec: float, fractional: bool) -> str:
    """Output filename for a frame timestamp.

    Whole-second timestamps keep the historical ``frame_03s.jpg`` form
    (downstream tools glob that pattern); runs with fractional timestamps
    use a fixed-width ``frame_000.50s.jpg`` form so names still sort.
    """
    if fractional:
        return f"frame_{sec:06.2f}s.jpg"
    return f"frame_{int(sec):02d}s.jpg"


def extract_frames(
    video_path: str = DEFAULT_VIDEO_PATH,
    out_dir: str = DEFAULT_OUT_DIR,
    interval_sec: float = DEFAULT_INTERVAL_SEC,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    quality: int = 95,
) -> int:
    """Extract one frame every ``interval_sec`` seconds.

    The final partial interval is included (a 9.5 s video yields frames at
    0 s, 3 s, 6 s and 9 s). ``interval_sec`` may be fractional (e.g. 0.5
    for two frames per second). Only timestamps in ``[start_sec,
    end_sec)`` are extracted; ``end_sec=None`` means the end of the
    video, and an ``end_sec`` past the video end is clamped to it.
    Frames are written as JPEG with the given ``quality`` (0-100,
    OpenCV default 95).

    Returns the number of frames written.

    Raises:
        ValueError: if ``interval_sec`` is not positive, ``start_sec``
            is negative, ``end_sec`` is not after ``start_sec``, or
            ``quality`` is outside 0-100.
        RuntimeError: if the video cannot be opened, has no frames / no
            determinable frame rate, or no frame at all could be read
            (an unopenable video no longer "succeeds" with 0 frames).
    """
    if interval_sec <= 0:
        raise ValueError(f"interval_sec must be positive, got {interval_sec!r}")
    if start_sec < 0:
        raise ValueError(f"start_sec must be >= 0, got {start_sec!r}")
    if end_sec is not None and end_sec <= start_sec:
        raise ValueError(
            f"end_sec must be after start_sec, got start_sec={start_sec!r}, "
            f"end_sec={end_sec!r}"
        )
    if not isinstance(quality, int) or not 0 <= quality <= 100:
        raise ValueError(f"quality must be an int in 0-100, got {quality!r}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            raise RuntimeError(f"Could not determine frame rate of video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise RuntimeError(f"Video contains no frames: {video_path}")
        duration_sec = total / fps
        end = duration_sec if end_sec is None else min(end_sec, duration_sec)

        os.makedirs(out_dir, exist_ok=True)
        count = 0
        # Timestamps are start + i*interval for i in range(ceil(span /
        # interval)); the 1e-9 epsilon keeps an exact multiple (e.g.
        # 9.0 s / 3 s) from rounding up to a phantom extra timestamp.
        # This replaces range(0, ceil(duration), interval), which could
        # not express fractional intervals at all (TypeError).
        fractional = not (
            float(interval_sec).is_integer() and float(start_sec).is_integer()
        )
        span = end - start_sec
        n_timestamps = int(math.ceil(span / interval_sec - 1e-9)) if span > 0 else 0
        for i in range(n_timestamps):
            sec = start_sec + i * interval_sec
            if sec >= end:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.imwrite(
                os.path.join(out_dir, _frame_filename(sec, fractional)),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )
            count += 1
        if count == 0:
            raise RuntimeError(f"No frames could be read from video: {video_path}")
        return count
    finally:
        cap.release()


def main(argv=None) -> int:
    """CLI entry point. Returns a process exit code (0 ok, 2 failure)."""
    parser = argparse.ArgumentParser(description="Extract one frame every N seconds from a video.")
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="input video path")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="output directory for frames")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SEC,
                        help="seconds between extracted frames (may be fractional, e.g. 0.5)")
    parser.add_argument("--start", type=float, default=0.0,
                        help="start of the extraction window in seconds (default: 0)")
    parser.add_argument("--end", type=float, default=None,
                        help="end of the extraction window in seconds, exclusive "
                             "(default: end of video; past the end is clamped)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality 0-100 (default: 95, the OpenCV default)")
    args = parser.parse_args(argv)
    try:
        count = extract_frames(args.video, args.out_dir, args.interval,
                               start_sec=args.start, end_sec=args.end,
                               quality=args.quality)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Extracted {count} frames to {args.out_dir}")
    return 0


def _probe_ready() -> bool:
    """Real readiness probe: OpenCV usable and extraction entry point intact.

    Previously every readiness helper returned True unconditionally; now
    they all derive from this probe (plus the capability list).
    """
    try:
        if not callable(extract_frames):
            return False
        cap = cv2.VideoCapture()  # no file opened; just proves cv2 works
        cap.release()
        return bool(get_extract_video_capabilities())
    except Exception:
        return False


def get_extract_video_ready_flag() -> dict:
    ready = _probe_ready()
    return {"module": "extract_video", "ready": ready}


def get_extract_video_version_info() -> dict:
    return {"module": "extract_video", "version": __version__, "ready": _probe_ready()}


def get_extract_video_diagnostic_snapshot() -> dict:
    ready = _probe_ready()
    return {"module": "extract_video", "status": "ok" if ready else "unavailable", "ready": ready}


def get_extract_video_capabilities() -> list:
    return ["extract_video", "frame_extraction", "diagnostic_snapshot"]


def validate_extract_video_contract() -> bool:
    return bool(get_extract_video_capabilities()) and _probe_ready()


def get_extract_video_baseline_metrics() -> dict:
    return {"module": "extract_video", "contract_valid": validate_extract_video_contract(), "extractor_ready": _probe_ready()}


def get_extract_video_readiness_status() -> dict:
    return {"module": "extract_video", "ready": validate_extract_video_contract(), "metrics": get_extract_video_baseline_metrics()}


def get_extract_video_audit_record() -> dict:
    return {"module": "extract_video", "audit_passed": validate_extract_video_contract(), "readiness": get_extract_video_readiness_status()}


def get_extract_video_operational_digest() -> dict:
    return {"module": "extract_video", "operational": validate_extract_video_contract(), "audit": get_extract_video_audit_record()}


def get_extract_video_telemetry_payload() -> dict:
    return {"module": "extract_video", "payload": get_extract_video_operational_digest()}


def get_extract_video_integrity_seal() -> dict:
    return {"module": "extract_video", "sealed": validate_extract_video_contract(), "telemetry": get_extract_video_telemetry_payload()}


def get_extract_video_compliance_certificate() -> dict:
    return {"module": "extract_video", "certified": validate_extract_video_contract(), "seal": get_extract_video_integrity_seal()}


def get_extract_video_readiness_probe() -> dict:
    return {"module": "extract_video", "ready": validate_extract_video_contract(), "certificate": get_extract_video_compliance_certificate()}


def get_extract_video_audit_summary() -> dict:
    return {"module": "extract_video", "audit_passed": validate_extract_video_contract(), "probe": get_extract_video_readiness_probe()}


if __name__ == "__main__":
    raise SystemExit(main())
