"""The real CV pipeline, wrapped for the API instead of Streamlit.

Loads the same components `app/streamlit_app.py`'s `get_pipeline_components()` does, as
module-level singletons (loaded once per process, not once per request -- these are
real PyTorch/Ultralytics models, reloading them per-request would be minutes of wasted
work on every single call). `run_analysis` mirrors `run_pipeline_with_timing` stage by
stage and returns a JSON-safe dict shaped to match the `video_analyses` Supabase table
this project's web app already reads from -- so a job's result can be inserted directly,
no reshaping at the call site.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_BALL_STUMPS_WEIGHTS = REPO_ROOT / "weights" / "ball_stumps_n.pt"


def _ball_stumps_weights_path() -> str | None:
    override = os.environ.get("THIRD_UMPIRE_BALL_STUMPS_WEIGHTS")
    if override:
        return override
    return str(_DEFAULT_BALL_STUMPS_WEIGHTS) if _DEFAULT_BALL_STUMPS_WEIGHTS.exists() else None


class _Components:
    """Lazy singleton so importing this module doesn't eagerly load PyTorch models --
    only the first real analysis request pays that cost, matching Streamlit's
    `@st.cache_resource` behaviour of "load once, on first use"."""

    _instance: dict[str, Any] | None = None

    @classmethod
    def get(cls) -> dict[str, Any]:
        if cls._instance is None:
            from video_engine.detection.yolo_detector import YoloDetector
            from video_engine.events.heuristic_segmenter import HeuristicEventSegmenter
            from video_engine.pose.keypoint_pose import KeypointRcnnPoseEstimator
            from video_engine.tracking.bytetrack_tracker import ByteTrackTracker

            cls._instance = {
                "detector": YoloDetector(ball_stumps_weights=_ball_stumps_weights_path()),
                "tracker": ByteTrackTracker(),
                "pose_estimator": KeypointRcnnPoseEstimator(),
                "event_segmenter": HeuristicEventSegmenter(),
            }
        return cls._instance


def probe_video(path: Path) -> tuple[int, int, float]:
    """(width, height, fps) via OpenCV -- same probe Streamlit's upload path relies on."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        return width, height, fps
    finally:
        cap.release()


def run_analysis(
    video_path: Path,
    source_label: str,
    match_id: str | None = None,
    innings: int | None = None,
    over: int | None = None,
    ball: int | None = None,
) -> dict[str, Any]:
    """Runs the real pipeline end to end and returns a JSON-safe result dict shaped for
    the `video_analyses` table: source_label, match_id/innings/over_number/ball_number,
    frames_processed, people_tracked, pose_frames, shot_classification, event_notes,
    detection_confidence. Raises on failure -- the caller (the job runner) is
    responsible for catching and recording that as a failed job, not this function
    papering over it."""
    from video_engine.contracts import DeliveryClip, DeliveryRef, Source
    from video_engine.io.clip_loader import load_frames

    width, height, fps = probe_video(video_path)
    clip = DeliveryClip(
        delivery=DeliveryRef(
            match_id=match_id or "api-upload",
            innings=innings or 1,
            over=over or 0,
            ball=ball or 1,
        ),
        video_path=str(video_path),
        fps=fps,
        width=width,
        height=height,
        source=Source.USER_UPLOAD,
    )

    components = _Components.get()

    t0 = time.perf_counter()
    frames = load_frames(clip)
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    detections = components["detector"].detect(frames)
    t_detect = time.perf_counter() - t0

    t0 = time.perf_counter()
    tracks = components["tracker"].track(detections)
    t_track = time.perf_counter() - t0

    t0 = time.perf_counter()
    poses = components["pose_estimator"].estimate(frames, tracks)
    t_pose = time.perf_counter() - t0

    t0 = time.perf_counter()
    event = components["event_segmenter"].segment(tracks, poses)
    t_event = time.perf_counter() - t0

    player_tracks = [t for t in tracks if t.obj_class.value == "player"]

    confidence: dict[str, dict[str, float | int]] = {}
    by_class: dict[str, list[float]] = {}
    for d in detections:
        by_class.setdefault(d.obj_class.value, []).append(d.confidence)
    for cls, values in by_class.items():
        confidence[cls] = {
            "count": len(values),
            "mean": round(sum(values) / len(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    return {
        "source_label": source_label,
        "match_id": match_id,
        "innings": innings,
        "over_number": over,
        "ball_number": ball,
        "frames_processed": len(frames),
        "people_tracked": len(player_tracks),
        "pose_frames": len(poses),
        "shot_classification": event.shot_type.value,
        "event_notes": event.notes or "",
        "detection_confidence": confidence,
        "timings": {
            "load_frames": round(t_load, 3),
            "detection": round(t_detect, 3),
            "tracking": round(t_track, 3),
            "pose_estimation": round(t_pose, 3),
            "event_segmentation": round(t_event, 3),
        },
    }
