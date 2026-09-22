"""How much decoded video this service will hold in memory at once, and why there has to
be a limit at all.

**The number that motivates this module.** The clip that killed the production container
is 13.1 MiB on disk: 3840x2160, 60fps, 9.97 seconds, 598 frames. `load_frames` decoded it
into a `list[np.ndarray]` of full-resolution RGB frames, and a 3840x2160x3 uint8 frame is
23.73 MiB. 598 of them is **13.86 GiB**, reached about 14 seconds into the job and then
held for the whole run while detection, tracking and pose estimation piled their own
allocations on top. That is a 1083x amplification between "the bytes the caller uploaded"
and "the bytes this process resident". Measured, not estimated.

This is why a byte-size upload limit does not bound anything that matters. Compression
ratio is a property of the codec and the footage, not of the request, and for modern
H.264/HEVC 4K it is three orders of magnitude. `THIRD_UMPIRE_MAX_UPLOAD_MB` defaults to
50, so the gate in front of this pipeline currently *accepts* clips whose decoded form
cannot fit in any machine this will ever run on. The limit has to be expressed in decoded
pixels, and it has to be checked before the job starts, not discovered 14 seconds in when
the kernel kills the process.

**Why downscaling costs almost nothing.** Neither model in this pipeline ever looks at a
4K frame. `YoloDetector` letterboxes to `imgsz=640` inside Ultralytics before its forward
pass. `KeypointRcnnPoseEstimator` uses torchvision's stock `KeypointRCNN_ResNet50_FPN`
transform, whose `min_size=800, max_size=1333` resizes 3840x2160 down to 1333x750 on the
way in. `overlay.DEFAULT_MAX_EDGE` caps returned sample frames at 960. So every consumer
of `frames` immediately shrinks what it is given to 1333 or less, and the other 13.8 GiB
of pixels are decoded, copied, held for the length of the job, and then thrown away
without any model having read them.

`DEFAULT_MAX_EDGE` is therefore 1333 rather than a round number: it is the largest edge
any consumer in this pipeline actually uses. Footage at or below it is passed through
untouched and its analysis is bit-identical to before. Footage above it is resized to the
size torchvision was going to resize it to anyway.

**What this does change, honestly.** Downscaling shrinks every box coordinate by the same
factor. Almost every spatial threshold in `video_engine` is already expressed as a
multiple of the ball box (`trajectory.fit.TOLERANCE_BOX_MULTIPLE`,
`MIN_DISPLACEMENT_BOX_MULTIPLE`, `MIN_STEP_SPEED_BOX_MULTIPLE`,
`geometry.CLUSTER_RADIUS_MULTIPLE`, `MIN_END_SIZE_RATIO`) and `motion.energy` works in
fractions of changed pixels, so all of those are scale-invariant and unaffected. Two
constants are absolute pixels and do shift meaning: `trajectory.fit.MIN_TOLERANCE_PX`
(4.0) and `calibration.MAX_REPROJECTION_ERROR_PX` (2.0). Both are floors/ceilings on
*error*, and error shrinks with the coordinates, so both become relatively more permissive
on downscaled footage -- the direction that accepts a fit rather than rejects one. That is
the safer direction to be wrong in, but it is a real difference and it is why
`run_analysis` reports the applied scale in its result instead of quietly resizing.

**Why frames are capped as well as resolution.** Resolution alone bounds bytes-per-frame;
it does not bound the number of frames, and duration is unbounded for a fixed upload size.
At `DEFAULT_MAX_EDGE` a 16:9 frame is about 3.0 MiB, so the frame cap is what turns
"bounded per frame" into "bounded, full stop".

**The frame cap is really a time limit.** Memory would allow far more than 900 frames --
900 of them is only about 2.6 GiB. What actually binds is Keypoint R-CNN. Measured end to
end on 48 real 4K frames through this exact path on a 6-thread CPU: 154.8 s total, of
which **142.9 s was pose estimation** -- 2.98 s per frame, and *independent of the edge
limit*, because torchvision's transform resizes whatever it is given to its own
`min_size=800` before the forward pass. Downscaling saves memory; it does not save pose
time. So the cap is set where the wait stays defensible: 900 frames is 15 seconds at
60fps or 36 at 25, which comfortably accepts the 598-frame clip from the incident while
keeping the worst case from becoming an hour-long job nobody is still polling for.

Whoever changes this should change it for that reason, and should expect a full 598-frame
4K clip to take roughly half an hour on six threads, proportionally less on the hosted
box's wider CPU. It is a slow pipeline; that is a known property (see README), not a
regression introduced here.

Over-length footage is **refused, not silently truncated**. Analysing the first 1200
frames of a longer clip and reporting the result as if it covered the delivery is the kind
of quiet wrong answer this project's MISTAKES.md exists to prevent -- a caller cannot tell
it happened, and "the ball was never released" and "the release was in the part we threw
away" produce the same empty answer.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# torchvision KeypointRCNN_ResNet50_FPN's transform max_size -- the largest edge anything
# downstream of load_frames actually consumes. See the module docstring.
DEFAULT_MAX_EDGE = 1333

# 15s at 60fps / 36s at 25fps. A delivery clip is a few seconds; this is slack, not a
# target. Bounded by pose-estimation time rather than by memory -- see the module
# docstring.
DEFAULT_MAX_FRAMES = 900

_BYTES_PER_PIXEL = 3  # uint8 RGB, what load_frames produces


class FrameBudgetError(ValueError):
    """The submitted footage cannot be decoded inside this service's memory budget.

    Carries a message written for the person who uploaded the clip, because it is
    returned to them verbatim as an HTTP 413 detail -- it has to say what was wrong with
    *their* file and what would work, not just name a limit.
    """


def max_edge() -> int:
    return int(os.environ.get("THIRD_UMPIRE_MAX_FRAME_EDGE", DEFAULT_MAX_EDGE))


def max_frames() -> int:
    return int(os.environ.get("THIRD_UMPIRE_MAX_FRAMES", DEFAULT_MAX_FRAMES))


@dataclass(frozen=True)
class FramePlan:
    """What decoding this clip will actually cost, decided before any of it is decoded."""

    source_width: int
    source_height: int
    source_frames: int
    decode_width: int
    decode_height: int
    scale: float
    frame_bytes: int
    total_bytes: int

    @property
    def downscaled(self) -> bool:
        return self.scale < 1.0

    def as_payload(self) -> dict[str, object]:
        """The block `run_analysis` attaches to every result.

        Present even when nothing was resized (`downscaled: false`), so a consumer reads
        one shape either way rather than inferring "absent means full resolution".
        """
        return {
            "downscaled": self.downscaled,
            "source_resolution": f"{self.source_width}x{self.source_height}",
            "analysed_resolution": f"{self.decode_width}x{self.decode_height}",
            "scale": round(self.scale, 4),
            "frames": self.source_frames,
            "decoded_mib": round(self.total_bytes / 1024 / 1024, 1),
            # Said out loud in the payload rather than left to the reader, because a
            # consumer comparing two runs of the same clip at different resolutions needs
            # to know whether the difference is meaningful.
            "note": (
                "Frames were downscaled to fit this service's memory budget. Both models "
                "resize below this edge internally, so detections are unaffected; box "
                "coordinates are in analysed-resolution pixels."
                if self.scale < 1.0
                else "Frames were analysed at their source resolution."
            ),
        }


def plan_decode(
    width: int,
    height: int,
    frame_count: int,
    *,
    edge_limit: int | None = None,
    frame_limit: int | None = None,
) -> FramePlan:
    """Decide the decode size for a clip, or refuse it.

    `frame_count` may be 0 or negative: `CAP_PROP_FRAME_COUNT` is unreliable for some
    containers and OpenCV reports 0 rather than failing. That is treated as "unknown, let
    the decoder find out" -- the count check is skipped here and `load_frames` enforces
    the same cap while decoding, so an unreliable probe cannot get past the memory bound,
    it only loses the fast rejection at submit time.

    Raises `FrameBudgetError` for footage that is too long, and for a clip whose
    dimensions OpenCV could not read at all (0x0), which means the file is not decodable
    video and is worth saying so immediately rather than starting a job that cannot work.
    """
    edge = edge_limit if edge_limit is not None else max_edge()
    limit = frame_limit if frame_limit is not None else max_frames()

    if width <= 0 or height <= 0:
        raise FrameBudgetError(
            "Could not read a frame size from this file. It is either not a video, or "
            "uses a codec this server cannot decode."
        )

    if frame_count > limit:
        raise FrameBudgetError(
            f"This clip is {frame_count} frames; this service analyses at most {limit} "
            f"in one job. Trim it to the delivery you want analysed and resubmit. "
            f"(Frames are held in memory to analyse, so the limit is on frames, not on "
            f"file size.)"
        )

    longest = max(width, height)
    scale = min(1.0, edge / longest)
    decode_width = max(1, int(round(width * scale)))
    decode_height = max(1, int(round(height * scale)))
    frame_bytes = decode_width * decode_height * _BYTES_PER_PIXEL
    counted = frame_count if frame_count > 0 else limit  # unknown: budget for the worst case

    return FramePlan(
        source_width=width,
        source_height=height,
        source_frames=max(frame_count, 0),
        decode_width=decode_width,
        decode_height=decode_height,
        scale=scale,
        frame_bytes=frame_bytes,
        total_bytes=frame_bytes * counted,
    )
