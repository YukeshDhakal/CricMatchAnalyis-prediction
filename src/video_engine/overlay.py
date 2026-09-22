"""Drawing real detections and poses onto real frames, and getting them out of the process.

This used to live inside `app/streamlit_app.py` as a private helper, which meant the only
way to *see* what the models saw was to run Streamlit on this machine. The API path needs
the same picture (the web app's demo role renders it), and importing a Streamlit module
from a FastAPI worker would drag Streamlit's whole runtime -- and its "no ScriptRunContext"
warnings -- into the server process for the sake of one `cv2.rectangle` loop. So the
drawing moved here, to the engine that owns the data being drawn, and both callers import
it from one place.

**Colour space.** `io.clip_loader.load_frames` converts every frame to RGB, so everything
in this module works in RGB and the box/keypoint colours below are RGB triples. That
matters at exactly one boundary: `cv2.imencode` expects BGR, so `encode_frame` converts on
the way out. Skipping that conversion does not fail, it just silently swaps red and blue in
every returned image -- which is why `tests/video_engine/test_overlay.py` decodes the
base64 back and asserts on actual pixel values rather than on the field being present.
"""
from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

# RGB, to match load_frames' output. Players and non-player objects are deliberately
# different hues so a frame with both is readable without a legend.
_PLAYER_COLOR = (255, 80, 80)
_OTHER_COLOR = (80, 220, 80)
_KEYPOINT_COLOR = (255, 230, 0)
_KEYPOINT_MIN_CONFIDENCE = 0.3

# A returned frame is a thumbnail for a web panel, not an archival asset: it is there to
# answer "is the model seeing a person or a hoarding?". Full-resolution broadcast frames
# base64-encode to roughly half a megabyte each, and four of those turn a job result into a
# multi-megabyte JSON body that has to cross Railway -> Vercel -> browser. Capping the long
# edge keeps the whole payload well under a megabyte while staying large enough to see a
# box and a skeleton. Boxes are drawn *before* the resize, so they scale with the image.
DEFAULT_MAX_EDGE = 960
DEFAULT_JPEG_QUALITY = 80
DEFAULT_SAMPLE_COUNT = 4


def draw_overlay(
    frame: np.ndarray,
    detections_at_frame: list,
    poses_at_frame: list,
) -> np.ndarray:
    """Draws real detection boxes (with confidence) and pose keypoints onto a real frame --
    the actual visual check of whether the model is seeing anything sensible, not just a
    count in a table.

    Takes and returns RGB. Does not mutate the frame it is given.
    """
    img = frame.copy()
    for d in detections_at_frame:
        x1, y1, x2, y2 = (int(v) for v in (d.box.x1, d.box.y1, d.box.x2, d.box.y2))
        color = _PLAYER_COLOR if d.obj_class.value == "player" else _OTHER_COLOR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{d.obj_class.value} {d.confidence:.2f}",
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    for pose in poses_at_frame:
        for kp in pose.keypoints:
            if kp.confidence > _KEYPOINT_MIN_CONFIDENCE:
                cv2.circle(img, (int(kp.x), int(kp.y)), 3, _KEYPOINT_COLOR, -1)
    return img


def _downscale(frame_rgb: np.ndarray, max_edge: int) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    longest = max(height, width)
    if max_edge <= 0 or longest <= max_edge:
        return frame_rgb
    scale = max_edge / float(longest)
    return cv2.resize(
        frame_rgb,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def encode_frame(
    frame_rgb: np.ndarray,
    max_edge: int = DEFAULT_MAX_EDGE,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> dict[str, Any]:
    """RGB frame -> a JSON-safe dict with a base64 JPEG and the dimensions of what's in it.

    The dimensions describe the *encoded* image (post-downscale), not the source frame, so
    a client can lay the image out without decoding it first, and so a test can tell a real
    picture from a 1x1 placeholder.
    """
    scaled = _downscale(frame_rgb, max_edge)
    bgr = cv2.cvtColor(scaled, cv2.COLOR_RGB2BGR)  # imencode is BGR-native; see docstring
    ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed to encode an overlay frame as JPEG.")
    height, width = scaled.shape[:2]
    return {
        "width": int(width),
        "height": int(height),
        "mime_type": "image/jpeg",
        "image_base64": base64.b64encode(buffer.tobytes()).decode("ascii"),
    }


def sample_overlay_frames(
    frames: list[np.ndarray],
    detections: list,
    poses: list,
    max_frames: int = DEFAULT_SAMPLE_COUNT,
    max_edge: int = DEFAULT_MAX_EDGE,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> list[dict[str, Any]]:
    """A handful of frames that actually contain detections, with overlays drawn on them.

    Samples the same way the Streamlit console does -- evenly across the frames that have
    at least one detection, rather than the first N, so a clip whose detections all cluster
    at the start doesn't return four near-identical pictures. Frames with no detections are
    never sampled: an empty frame proves nothing and costs the same bytes.

    Returns `[]` when nothing was detected anywhere. That is a real, honest answer (the
    clip had nothing in it the model recognised) and the caller renders it as such rather
    than as a broken image.
    """
    detected_indices = sorted({d.frame_index for d in detections})
    if not detected_indices or max_frames <= 0:
        return []

    step = max(1, len(detected_indices) // max_frames)
    chosen = detected_indices[::step][:max_frames]

    out: list[dict[str, Any]] = []
    for frame_index in chosen:
        if frame_index < 0 or frame_index >= len(frames):
            continue  # a detection pointing outside the decoded frames is a bug elsewhere
        dets_here = [d for d in detections if d.frame_index == frame_index]
        poses_here = [p for p in poses if p.frame_index == frame_index]
        overlay = draw_overlay(frames[frame_index], dets_here, poses_here)
        encoded = encode_frame(overlay, max_edge=max_edge, quality=quality)
        encoded["frame_index"] = int(frame_index)
        encoded["detections"] = len(dets_here)
        encoded["poses"] = len(poses_here)
        out.append(encoded)
    return out


def tracks_payload(tracks: list) -> list[dict[str, Any]]:
    """The per-track table the Streamlit console shows: id, class, how many frames it was
    seen in. Sorted by class then id so the ordering is stable between runs of the same
    clip rather than following tracker internals."""
    return sorted(
        (
            {
                "track_id": int(t.track_id),
                "obj_class": t.obj_class.value,
                "frames_seen": len(t.detections),
            }
            for t in tracks
        ),
        key=lambda row: (row["obj_class"], row["track_id"]),
    )
