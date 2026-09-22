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


def _pitch_geometry_payload(event: Any) -> dict[str, Any]:
    """The line/length block of the API response, with its uncertainty attached.

    Shaped so the *only* safe way to read it is to check `available` first. A response
    that carried `pitch_length: "good"` at the top level would be read as a fact by any
    client that forgot to look for a confidence field next to it, and on single-camera
    footage it is never a fact -- it is an estimate resting on a stump-end assumption and
    an 88:1 calibration rectangle. `insufficient_data_reason` is populated whenever
    `available` is false so a user sees why rather than assuming the feature is broken.
    """
    from video_engine.contracts import GeometryConfidence

    trustworthy = event.pitch_confidence in (
        GeometryConfidence.MEDIUM,
        GeometryConfidence.HIGH,
    )
    payload: dict[str, Any] = {
        "available": bool(event.pitch_point is not None and trustworthy),
        "confidence": event.pitch_confidence.value,
        "track_confidence": round(float(event.track_confidence), 3),
        "notes": event.pitch_notes or "",
        "bounce_frame": event.bounce_frame,
    }
    if event.pitch_point is None:
        payload["insufficient_data_reason"] = event.pitch_notes or "No pitch geometry was produced."
        return payload
    payload["pitch_length_band"] = event.pitch_length.value
    payload["pitch_length_m"] = event.pitch_point.length_m
    # Signed distance from the middle stump, and deliberately unlabelled: which side is
    # off and which is leg depends on the striker's handedness, which this project has no
    # source for. See contracts.PitchPoint.
    payload["pitch_line_m"] = event.pitch_point.line_m
    payload["line_side_labelled"] = False
    if not trustworthy:
        payload["insufficient_data_reason"] = (
            "A bounce point was projected but its confidence is "
            f"'{event.pitch_confidence.value}'; treat it as indicative only."
        )
    return payload


def run_analysis(
    video_path: Path,
    source_label: str,
    match_id: str | None = None,
    innings: int | None = None,
    over: int | None = None,
    ball: int | None = None,
    include_frames: bool = False,
) -> dict[str, Any]:
    """Runs the real pipeline end to end and returns a JSON-safe result dict shaped for
    the `video_analyses` table: source_label, match_id/innings/over_number/ball_number,
    frames_processed, people_tracked, pose_frames, shot_classification, event_notes,
    detection_confidence. Raises on failure -- the caller (the job runner) is
    responsible for catching and recording that as a failed job, not this function
    papering over it.

    `include_frames` additionally attaches `tracks` (the per-track table) and
    `sample_frames` (a few real frames with real boxes and keypoints drawn on them,
    base64 JPEG). It defaults to False because those frames are the only part of this
    response whose size depends on the footage rather than on a fixed set of scalars --
    a handful of encoded frames is several hundred kilobytes, against roughly one for
    everything else. A caller that isn't going to display them shouldn't pay to move
    them, and the columns of the `video_analyses` table have nowhere to put them either.
    """
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
    event = components["event_segmenter"].segment(tracks, poses, frames)
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

    result: dict[str, Any] = {
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
        # Pitch geometry always travels with the evidence behind it. `pitch_length` and
        # `pitch_line_m` are meaningless without `pitch_confidence`, so they are nested
        # together rather than sitting as sibling top-level keys a consumer could read
        # one of and not the other. `available` is the single field a client should
        # branch on before displaying anything here.
        "pitch_geometry": _pitch_geometry_payload(event),
        "timings": {
            "load_frames": round(t_load, 3),
            "detection": round(t_detect, 3),
            "tracking": round(t_track, 3),
            "pose_estimation": round(t_pose, 3),
            "event_segmentation": round(t_event, 3),
        },
    }

    if include_frames:
        # Imported here, not at module scope, so a caller that never asks for frames
        # doesn't pay the import -- and so this module keeps its "nothing heavy until
        # the first real request" property.
        from video_engine.overlay import sample_overlay_frames, tracks_payload

        result["tracks"] = tracks_payload(tracks)
        # Empty when nothing was detected anywhere in the clip. That is a real answer,
        # not a failure: the caller renders "no detections" rather than broken images.
        result["sample_frames"] = sample_overlay_frames(frames, detections, poses)

    return result
