"""Reads the video file ingestion hands off into frames the rest of the engine works on.

**Decoding video is a memory amplifier, and the amplification is enormous.** A 13.1 MiB
H.264 file at 3840x2160/60fps holds 598 frames; decoded to uint8 RGB that is 23.73 MiB per
frame and **13.86 GiB** for the list -- 1083x the file on disk. Measured on real footage,
after that exact clip killed a production container. Nothing about the file's size hints
at this: the ratio is a property of the codec and the content.

So `max_edge` and `max_frames` exist, and any caller that decodes footage it did not
choose itself -- anything reachable from a network -- should pass them. They default to
`None` (decode everything, exactly as before) because this function is the engine's
faithful decoder and the local callers that hand it a file they picked are entitled to
full resolution. `api.frame_budget` is where the hosted service's limits are decided and
justified; this module only enforces what it is told.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..contracts import DeliveryClip


class ClipTooLongError(ValueError):
    """The clip has more frames than the caller's budget allows.

    Raised rather than returning a truncated list. A short read would be indistinguishable
    from a short clip at every call site, and "no ball was released" and "the release was
    in the frames we dropped" are the same answer to a caller who cannot tell them apart.
    """


def load_frames(
    clip: DeliveryClip,
    *,
    max_edge: int | None = None,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    """Decode `clip` to a list of uint8 RGB frames.

    `max_edge` downscales any frame whose longest edge exceeds it, **during decode**, so
    the full-resolution buffer is never retained -- peak memory is the (bounded) frame
    list plus the single frame being converted, not the list at source resolution.
    `INTER_AREA` is the resampling filter because this is always a downscale and it is the
    one that averages rather than samples: a cricket ball is a handful of pixels and
    point-sampling it through a 3x reduction can drop it entirely between frames.

    `max_frames` raises `ClipTooLongError` on the frame *after* the limit, so a clip
    exactly at the limit is accepted.
    """
    capture = cv2.VideoCapture(clip.video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open delivery clip: {clip.video_path}")

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if max_frames is not None and len(frames) >= max_frames:
                raise ClipTooLongError(
                    f"Clip has more than {max_frames} frames; this caller decodes at most "
                    f"{max_frames}."
                )
            if max_edge is not None:
                longest = max(frame.shape[0], frame.shape[1])
                if longest > max_edge:
                    scale = max_edge / longest
                    # Resize before the colour convert: both are O(pixels) and doing the
                    # cheap one second is free. They commute -- INTER_AREA averages each
                    # channel independently, so the result is identical either way.
                    frame = cv2.resize(
                        frame,
                        (
                            max(1, int(round(frame.shape[1] * scale))),
                            max(1, int(round(frame.shape[0] * scale))),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames
