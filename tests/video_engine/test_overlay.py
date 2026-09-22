"""Tests that the overlay images are actually images of the right thing.

The failure mode this file exists for is not "the field is missing". It is "the field is
present, the JSON validates, the client renders an `<img>`, and the picture is wrong" --
red and blue swapped because `cv2.imencode` wants BGR while every frame in this engine is
RGB, or a 1x1 placeholder, or a frame with no boxes on it. None of that raises. All of it
is invisible to a test that only asserts `"image_base64" in result`.

So every test here decodes the base64 back into pixels and asserts on the pixels.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from video_engine.contracts import BoundingBox, Detection, Keypoint, ObjectClass, PoseFrame, Track
from video_engine.overlay import (
    draw_overlay,
    encode_frame,
    sample_overlay_frames,
    tracks_payload,
)


def _blank_frame(width: int = 320, height: int = 240, color=(0, 0, 0)) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def _decode(encoded: dict) -> np.ndarray:
    """base64 JPEG -> RGB array, i.e. exactly what a browser would end up showing."""
    raw = base64.b64decode(encoded["image_base64"])
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert bgr is not None, "the returned base64 did not decode as an image at all"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _detection(frame_index: int, obj_class=ObjectClass.PLAYER, box=(40, 40, 140, 200)) -> Detection:
    return Detection(
        frame_index=frame_index,
        obj_class=obj_class,
        box=BoundingBox(*(float(v) for v in box)),
        confidence=0.87,
    )


def test_draw_overlay_does_not_mutate_the_source_frame():
    frame = _blank_frame()
    original = frame.copy()
    draw_overlay(frame, [_detection(0)], [])
    assert np.array_equal(frame, original), "draw_overlay drew on the caller's frame"


def test_draw_overlay_actually_draws_a_box_where_the_detection_is():
    frame = _blank_frame()
    overlay = draw_overlay(frame, [_detection(0, box=(40, 40, 140, 200))], [])

    # The box edge is red-ish (255, 80, 80) in RGB. Sample a pixel on the left edge,
    # mid-height, rather than a corner, so line joins/thickness don't matter.
    edge_pixel = overlay[120, 40]
    assert edge_pixel[0] > 150, f"left box edge is not red-dominant: {edge_pixel}"
    assert edge_pixel[0] > edge_pixel[2], f"red/blue look swapped: {edge_pixel}"

    # And the inside of the box is untouched -- a filled rectangle would mean the box is
    # hiding the very footage it is meant to annotate.
    assert tuple(overlay[120, 90]) == (0, 0, 0)


def test_draw_overlay_plots_confident_keypoints_and_skips_unconfident_ones():
    frame = _blank_frame()
    pose = PoseFrame(
        frame_index=0,
        track_id=1,
        keypoints=[
            Keypoint(name="nose", x=200.0, y=100.0, confidence=0.9),
            Keypoint(name="left_ankle", x=260.0, y=100.0, confidence=0.1),
        ],
    )
    overlay = draw_overlay(frame, [], [pose])

    drawn = overlay[100, 200]
    assert drawn[0] > 150 and drawn[1] > 150 and drawn[2] < 100, f"keypoint not yellow: {drawn}"
    assert tuple(overlay[100, 260]) == (0, 0, 0), "a 0.1-confidence keypoint was drawn"


def test_encode_frame_round_trips_colours_rather_than_swapping_them():
    """The BGR/RGB trap, asserted directly: encode a pure-red RGB frame and check the
    decoded image is still red, not blue."""
    red = _blank_frame(width=64, height=48, color=(255, 0, 0))
    decoded = _decode(encode_frame(red, max_edge=0))

    mean = decoded.reshape(-1, 3).mean(axis=0)
    assert mean[0] > 200, f"red channel lost in the round trip: {mean}"
    assert mean[2] < 60, f"red and blue are swapped: {mean}"


def test_encode_frame_reports_the_dimensions_of_what_it_actually_encoded():
    frame = _blank_frame(width=1920, height=1080)
    encoded = encode_frame(frame, max_edge=960)

    decoded = _decode(encoded)
    assert decoded.shape[:2] == (encoded["height"], encoded["width"])
    assert max(decoded.shape[:2]) == 960, "max_edge was not applied"
    assert decoded.shape[0] / decoded.shape[1] == pytest.approx(1080 / 1920, abs=0.01)
    assert encoded["mime_type"] == "image/jpeg"


def test_encode_frame_leaves_small_frames_alone():
    encoded = encode_frame(_blank_frame(width=320, height=240), max_edge=960)
    assert (encoded["width"], encoded["height"]) == (320, 240)


def test_sample_overlay_frames_returns_real_pictures_with_the_boxes_on_them():
    frames = [_blank_frame() for _ in range(10)]
    detections = [_detection(i) for i in (1, 3, 5, 7, 9)]
    poses = [
        PoseFrame(frame_index=1, track_id=1, keypoints=[Keypoint("nose", 200.0, 100.0, 0.9)])
    ]

    sampled = sample_overlay_frames(frames, detections, poses, max_frames=4)

    assert len(sampled) == 4
    assert [s["frame_index"] for s in sampled] == [1, 3, 5, 7]
    assert all(s["detections"] == 1 for s in sampled)
    assert sampled[0]["poses"] == 1 and sampled[1]["poses"] == 0

    for s in sampled:
        decoded = _decode(s)
        assert decoded.shape[:2] == (240, 320), "sampled frame is not the size it claims"
        # Non-trivial: a blank frame with a box drawn on it has more than one distinct
        # colour. This is the check that catches an all-black or 1x1 placeholder.
        assert len(np.unique(decoded.reshape(-1, 3), axis=0)) > 1
        assert decoded[120, 40][0] > 120, "the box is missing from the encoded image"


def test_sample_overlay_frames_only_samples_frames_that_have_detections():
    frames = [_blank_frame() for _ in range(6)]
    sampled = sample_overlay_frames(frames, [_detection(4)], [], max_frames=4)
    assert [s["frame_index"] for s in sampled] == [4]


def test_sample_overlay_frames_is_empty_when_nothing_was_detected():
    assert sample_overlay_frames([_blank_frame()], [], []) == []


def test_sample_overlay_frames_ignores_a_detection_past_the_end_of_the_clip():
    """A detection index outside the decoded frames is a bug somewhere upstream; it must
    not become an IndexError that fails an otherwise-good job."""
    frames = [_blank_frame() for _ in range(3)]
    sampled = sample_overlay_frames(frames, [_detection(1), _detection(99)], [])
    assert [s["frame_index"] for s in sampled] == [1]


def test_tracks_payload_counts_frames_seen_and_orders_stably():
    tracks = [
        Track(track_id=7, obj_class=ObjectClass.PLAYER, detections=[_detection(0), _detection(1)]),
        Track(track_id=2, obj_class=ObjectClass.BALL, detections=[_detection(0, ObjectClass.BALL)]),
        Track(track_id=1, obj_class=ObjectClass.PLAYER, detections=[_detection(3)]),
    ]
    assert tracks_payload(tracks) == [
        {"track_id": 2, "obj_class": "ball", "frames_seen": 1},
        {"track_id": 1, "obj_class": "player", "frames_seen": 1},
        {"track_id": 7, "obj_class": "player", "frames_seen": 2},
    ]


def test_tracks_payload_is_empty_for_no_tracks():
    assert tracks_payload([]) == []
