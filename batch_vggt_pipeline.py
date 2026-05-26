"""
Fetch Studio X Lowes scan-artifacts JSON (optional), download the scan video, extract frames,
and run VGGT-SLAM (main.py).

Use :func:`download_studiox_scan_video` when starting from an API URL; use
:func:`run_studiox_scan_pipeline` with a local video path when the file is already on disk.

Frames are sampled with OpenCV (``cv2``) at a fixed output frame rate.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import requests

REPO_ROOT = Path(__file__).resolve().parent

# Default thresholds for main.py
_DEFAULT_CONF_THRESHOLD = 15.0
_DEFAULT_MIN_DISPARITY = 10.0
_FPS = 2.0
# Second attempt after main.py fails on the default thresholds.
_SLAM_RETRY_CONF_THRESHOLD = 50.0
_SLAM_RETRY_MIN_DISPARITY = 20.0
_DEFAULT_IMAGE_RESOLUTION = 512


def _run_slam_subprocess(
    main_py: Path,
    images_dir: Path,
    out_dir: Path,
    project_id: str,
    *,
    conf_threshold: float,
    min_disparity: float,
    image_resolution: int = _DEFAULT_IMAGE_RESOLUTION,
) -> None:
    slam = subprocess.run(
        [
            "python3",
            str(main_py),
            "--image_resolution",
            str(image_resolution),
            "--image_folder",
            str(images_dir),
            "--use_all_frames",
            "--log_results",
            "--skip_dense_log",
            "--min_disparity",
            str(min_disparity),
            "--conf_threshold",
            str(conf_threshold),
            "--log_path",
            str(out_dir / f"{project_id}_poses.txt"),
            #"--vis_map",
            "--run_os",
        ],
        cwd=str(REPO_ROOT),
    )
    if slam.returncode != 0:
        raise RuntimeError(f"main.py exited with code {slam.returncode}")


def _project_id_from_scan_artifacts_url(url: str) -> str:
    parsed = urlparse(url)
    q = parse_qs(parsed.query)
    ids = q.get("projectId") or []
    if not ids or not ids[0]:
        raise ValueError("URL must include a projectId query parameter, e.g. ...?projectId=PRJ-XXXX")
    return ids[0]


def _first_video_url(payload: dict[str, Any]) -> str:
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("API response has no data entries")
    for item in data:
        if not isinstance(item, dict):
            continue
        v = item.get("video")
        if isinstance(v, str) and v.strip():
            return v.strip()
    raise ValueError("No non-empty video URL in data")


def _extract_frames_cv2(
    video_file: Path,
    out_dir: Path,
    fps_out: float,
    *,
    start_s: float = 0.0,
) -> list[Path]:
    """
    Sample frames from ``video_file`` at ``fps_out`` Hz, writing ``000000.png``, …
    Timestamps are ``start_s + k / fps_out``; source frame index is
    ``round(t * video_fps)`` (same idea as ``_extract_frame_batch``).
    """
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_file}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if video_fps <= 0:
        video_fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count > 0:
        max_time = start_s + (frame_count - 1) / video_fps
    else:
        max_time = float("inf")

    written: list[Path] = []
    k = 0
    while True:
        t_v = start_s + k / fps_out
        if frame_count > 0 and t_v > max_time + 1e-6:
            break

        idx = int(round(t_v * video_fps))
        if frame_count > 0:
            idx = min(max(idx, 0), frame_count - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            if not written:
                raise RuntimeError(f"No frames read from {video_file}")
            break

        fname = out_dir / f"{k:06d}.png"
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(str(fname), frame)
        written.append(fname)
        k += 1

    cap.release()
    if not written:
        raise RuntimeError(f"No frames extracted from {video_file}")
    return written


def download_studiox_scan_video(
    api_url: str,
    *,
    base_output_dir: Path | None = None,
) -> tuple[Path, Path, str]:
    """
    GET ``api_url`` (scan-artifacts JSON), resolve ``projectId`` from the query string,
    create ``{timestamp}_{projectId}/`` under ``base_output_dir``, and download the first
    ``data[*].video`` URL into that folder.

    Parameters
    ----------
    api_url
        e.g. ``https://api.studioxlowes.com/spatial/v1/scan-artifacts?projectId=PRJ-Q2W9SL836``
    base_output_dir
        Parent directory for ``{timestamp}_{projectId}``. Defaults to the current working directory.

    Returns
    -------
    output_dir
        ``{timestamp}_{projectId}/`` directory that will contain the video (and typically ``images/``).
    video_path
        Path to the downloaded video file.
    project_id
        From the ``projectId`` query parameter on ``api_url``.
    """
    project_id = _project_id_from_scan_artifacts_url(api_url)
    parent = (base_output_dir or Path.cwd()).resolve()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = parent / f"{ts}_{project_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    r = requests.get(api_url, timeout=120)
    r.raise_for_status()
    payload = r.json()
    video_url = _first_video_url(payload)
    print(video_url)

    video_name = Path(urlparse(video_url).path).name or "scan_video.mp4"
    video_path = out_dir / video_name
    with requests.get(video_url, stream=True, timeout=600) as vr:
        vr.raise_for_status()
        with open(video_path, "wb") as f:
            for chunk in vr.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return out_dir, video_path, project_id


def run_studiox_scan_pipeline(
    video_path: Path | str,
    project_id: str,
    *,
    output_dir: Path | None = None,
    extract_fps: float = _FPS,
    skip_slam: bool = False,
    conf_threshold: float = _DEFAULT_CONF_THRESHOLD,
    min_disparity: float = _DEFAULT_MIN_DISPARITY,
    image_resolution: int = _DEFAULT_IMAGE_RESOLUTION,
) -> dict[str, Path | str | bool]:
    """
    1. Ensure ``output_dir`` exists (default: parent directory of ``video_path``).
    2. Extract frames at ``extract_fps`` into ``images/`` under that folder.
    3. Run ``main.py`` from the repo root with ``--vis_map``, logging, and SLAM hyperparameters
       (see implementation for the full argument list). Poses and related logs are written under
       the project folder via ``--log_path <project_dir>/poses.txt``.

    Use :func:`download_studiox_scan_video` when you only have a scan-artifacts API URL;
    when you already have a local video file, call this function directly.

    Parameters
    ----------
    video_path
        Path to the scan video (e.g. ``.mp4``).
    project_id
        Used for log naming and ``main.py`` output; must match your project when reproducing
        Studio X runs (e.g. from the API URL query string).
    output_dir
        Working folder for ``images/`` and SLAM outputs. Defaults to ``video_path``'s parent.
    extract_fps
        Target sampling rate in Hz for frame extraction with OpenCV (default 2).
    skip_slam
        If True, only extract frames (no ``main.py``).
    conf_threshold, min_disparity
        Passed to ``main.py``. If the first SLAM run fails, one retry uses
        ``conf_threshold`` 50 and ``min_disparity`` 20.
    image_resolution
        Passed to ``main.py`` (default 512; must be divisible by 16).

    Returns
    -------
    dict with keys: ``output_dir``, ``video_path``, ``images_dir``, ``project_id``,
    and ``slam_retried`` (True if the first SLAM run failed and the retry succeeded).
    """
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = (output_dir or video_path.parent).resolve()
    images_dir = out_dir / "images"
    # INSERT_YOUR_CODE
    if images_dir.exists():
        # Remove the images_dir and all its contents before creating a new one
        import shutil
        shutil.rmtree(images_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    _extract_frames_cv2(video_path, images_dir, extract_fps)

    slam_retried = False
    if not skip_slam:
        main_py = REPO_ROOT / "main.py"
        if not main_py.is_file():
            raise FileNotFoundError(f"main.py not found at {main_py}")

        try:
            _run_slam_subprocess(
                main_py,
                images_dir,
                out_dir,
                project_id,
                conf_threshold=conf_threshold,
                min_disparity=min_disparity,
                image_resolution=image_resolution,
            )
        except RuntimeError as e:
            print(
                f"SLAM failed ({e}); retrying with "
                f"conf_threshold={_SLAM_RETRY_CONF_THRESHOLD}, "
                f"min_disparity={_SLAM_RETRY_MIN_DISPARITY} …",
                flush=True,
            )
            _run_slam_subprocess(
                main_py,
                images_dir,
                out_dir,
                project_id,
                conf_threshold=_SLAM_RETRY_CONF_THRESHOLD,
                min_disparity=_SLAM_RETRY_MIN_DISPARITY,
                image_resolution=image_resolution,
            )
            slam_retried = True

    return {
        "output_dir": out_dir,
        "video_path": video_path,
        "images_dir": images_dir,
        "project_id": project_id,
        "slam_retried": slam_retried,
    }


def _iter_scan_artifact_urls(links_file: Path) -> list[str]:
    """Return non-empty, non-``#`` comment lines from a text file."""
    if not links_file.is_file():
        raise FileNotFoundError(f"Links file not found: {links_file}")
    out: list[str] = []
    for line in links_file.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    if not out:
        raise ValueError(f"No scan-artifact URLs in {links_file}")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python3 batch_vggt_pipeline.py <links.txt>",
            file=sys.stderr,
        )
        sys.exit(1)
    links_path = Path(sys.argv[1]).expanduser().resolve()
    urls = _iter_scan_artifact_urls(links_path)
    succeeded_first: list[str] = []
    succeeded_after_retry: list[str] = []
    failed: list[str] = []

    for i, url in enumerate(urls):
        print(f"[{i + 1}/{len(urls)}] {url}", flush=True)
        try:
            _, video_path, project_id = download_studiox_scan_video(url)
            result = run_studiox_scan_pipeline(video_path, project_id)
            print(result, flush=True)
            if result.get("slam_retried"):
                succeeded_after_retry.append(url)
            else:
                succeeded_first.append(url)
        except Exception as e:
            failed.append(url)
            print(f"Failed: {e}", file=sys.stderr, flush=True)

    print("\n=== Summary ===", flush=True)
    print(
        f"Succeeded on first try ({len(succeeded_first)}):",
        flush=True,
    )
    for u in succeeded_first:
        print(f"  {u}", flush=True)
    print(
        f"\nSucceeded after SLAM retry "
        f"(conf_threshold={_SLAM_RETRY_CONF_THRESHOLD}, "
        f"min_disparity={_SLAM_RETRY_MIN_DISPARITY}) ({len(succeeded_after_retry)}):",
        flush=True,
    )
    for u in succeeded_after_retry:
        print(f"  {u}", flush=True)
    print(f"\nFailed ({len(failed)}):", flush=True)
    for u in failed:
        print(f"  {u}", flush=True)

    if failed:
        sys.exit(1)
