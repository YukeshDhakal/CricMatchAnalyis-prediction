"""Tests that the decoder actually honours its memory bound, against real video files.

Not synthetic frame lists: the bug being fixed here lives entirely in the step between a
file on disk and a list of arrays, so a test that starts from arrays cannot see it. These
write real video with OpenCV's own encoder and decode it back, and assert on `nbytes` --
the number that grew to 13.86 GiB in production -- rather than on a frame count.

One test runs against the actual 4K clip from the incident when
`THIRD_UMPIRE_REAL_4K_CLIP` points at it, and skips otherwise. It is skipped in CI and on
any machine without the file, which is why the synthetic ones carry the same assertions:
the real clip is confirmation, not the only evidence.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from video_engine.contracts import DeliveryClip, DeliveryRef, Source
from video_engine.io.clip_loader import ClipTooLongError, load_frames

_MIB = 1024 * 1024


def _write_clip(path, width, height, frames):
    """A real encoded file. MJPG/AVI because OpenCV can write it without an external
    ffmpeg binary, which this machine does not have on PATH (see MISTAKES.md)."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (width, height)
    )
    assert writer.isOpened(), "OpenCV could not open a writer for this fixture"
    rng = np.random.default_rng(0)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
    writer.release()
    return path


def _clip(path, width, height):
    return DeliveryClip(
        delivery=DeliveryRef(match_id="t", innings=1, over=0, ball=1),
        video_path=str(path),
        fps=25.0,
        width=width,
        height=height,
        source=Source.USER_UPLOAD,
    )


def _payload_mib(frames):
    return sum(f.nbytes for f in frames) / _MIB


def test_an_unbounded_decode_is_the_bug(tmp_path):
    """Establishes the baseline the bound is measured against.

    Twenty 1080p frames is 118 MiB from a file of a few megabytes. The production clip was
    598 frames at four times the area; the arithmetic from here to 13.86 GiB is just
    multiplication, and nothing in the decoder objected to any of it.
    """
    path = _write_clip(tmp_path / "hd.avi", 1920, 1080, 20)
    frames = load_frames(_clip(path, 1920, 1080))

    assert len(frames) == 20
    assert _payload_mib(frames) == pytest.approx(20 * 1920 * 1080 * 3 / _MIB, rel=0.01)
    assert _payload_mib(frames) > 100


def test_max_edge_bounds_what_is_held(tmp_path):
    path = _write_clip(tmp_path / "hd.avi", 1920, 1080, 20)

    unbounded = load_frames(_clip(path, 1920, 1080))
    bounded = load_frames(_clip(path, 1920, 1080), max_edge=480)

    assert len(bounded) == len(unbounded)
    assert bounded[0].shape[1] == 480
    assert bounded[0].shape[0] == 270  # aspect ratio preserved
    # 4x on each edge is 16x the bytes, and that ratio is the entire fix.
    assert _payload_mib(unbounded) / _payload_mib(bounded) == pytest.approx(16, rel=0.05)


def test_frames_below_the_edge_limit_are_not_touched(tmp_path):
    """No resize, no resample, no difference. Footage that already fits must come back
    bit-identical to what an unbounded decode produced, or the bound has quietly changed
    the analysis of every clip that was working fine."""
    path = _write_clip(tmp_path / "small.avi", 320, 240, 5)

    unbounded = load_frames(_clip(path, 320, 240))
    bounded = load_frames(_clip(path, 320, 240), max_edge=1333)

    assert len(bounded) == len(unbounded)
    for a, b in zip(unbounded, bounded):
        assert np.array_equal(a, b)


def test_the_colour_channels_survive_the_resize(tmp_path):
    """`load_frames` returns RGB and now resizes on the way. Resizing a BGR frame and
    converting, versus converting and resizing, must not be the moment red and blue swap
    -- this repo has already shipped within one commit of that bug once (MISTAKES.md)."""
    path = tmp_path / "red.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (640, 480))
    red_bgr = np.zeros((480, 640, 3), np.uint8)
    red_bgr[:, :, 2] = 255  # BGR: red
    for _ in range(3):
        writer.write(red_bgr)
    writer.release()

    frames = load_frames(_clip(path, 640, 480), max_edge=160)

    assert frames[0].shape[1] == 160
    mean = frames[0].reshape(-1, 3).mean(axis=0)
    assert mean[0] > 200, f"red channel should dominate in RGB output, got {mean}"
    assert mean[2] < 60, f"blue channel should be near zero, got {mean}"


def test_too_many_frames_raises_rather_than_returning_a_short_list(tmp_path):
    """A truncated decode is indistinguishable from a short clip at every call site. "The
    ball was never released" and "the release was in the frames we dropped" would be the
    same answer, and only one of them is true."""
    path = _write_clip(tmp_path / "long.avi", 320, 240, 12)

    with pytest.raises(ClipTooLongError, match="more than 5 frames"):
        load_frames(_clip(path, 320, 240), max_frames=5)


def test_a_clip_exactly_at_the_frame_limit_is_accepted(tmp_path):
    path = _write_clip(tmp_path / "exact.avi", 320, 240, 8)
    frames = load_frames(_clip(path, 320, 240), max_frames=8)
    assert len(frames) == 8


@pytest.mark.skipif(
    not os.environ.get("THIRD_UMPIRE_REAL_4K_CLIP"),
    reason="set THIRD_UMPIRE_REAL_4K_CLIP to the 4K clip from the incident to run this",
)
def test_the_real_4k_incident_clip_decodes_inside_the_budget():
    """The actual file that killed the container, decoded end to end on this machine.

    Every one of its 598 frames is decoded -- nothing is skipped, because skipping frames
    would change the analysis and is not what the bound does. The edge limit is tightened
    to 320 rather than the shipped 1333 only so the test fits on a laptop: at the shipped
    bound this same clip is about 1.7 GiB, which is comfortable in a 24 GB container and
    not in a test run. The ratio is what is being asserted, and it is the same ratio.
    """
    path = os.environ["THIRD_UMPIRE_REAL_4K_CLIP"]
    capture = cv2.VideoCapture(path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert (width, height) == (3840, 2160), "expected the 4K clip from the incident"

    frames = load_frames(_clip(path, width, height), max_edge=320, max_frames=1200)

    assert len(frames) == source_frames, "every frame is still analysed"
    assert max(frames[0].shape[:2]) == 320
    unbounded_gib = source_frames * width * height * 3 / 1024**3
    assert round(unbounded_gib, 1) == 13.9, "this is the clip that filled the container"
    # The same 598 frames, held at a bounded edge: three orders of magnitude less.
    assert _payload_mib(frames) < 120
    assert unbounded_gib * 1024 / _payload_mib(frames) > 100
