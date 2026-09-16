"""Tests for the detector's declared class maps (`video_engine.detection.yolo_detector`).

These cover the *licensing/capability compromise*, not inference. The compromise is
recorded in the README's "Ball and stumps detection": no fine-tuned ball/stumps
checkpoint ships here, none is downloaded, and the only genuinely fine-tuned cricket-ball
detector found in public is single-class and carries no licence file (so it is
all-rights-reserved and cannot be used without the author's permission). The consequence
for this module is that a *ball-only* checkpoint is the realistic near-term option, and
a ball-only checkpoint must not be able to masquerade as a ball-and-stumps one.

That is what the two named class maps are for, and it is what is pinned here: loading a
one-class model under the two-class map would emit correct BALL detections and silently
never emit STUMPS, which is indistinguishable from "no stumps were visible in this clip"
and would leave pitch calibration quietly disabled with no error anywhere.

Only the maps are tested. `YoloDetector.__init__` loads real model weights (downloading
them on first use), so instantiating one is an integration concern and belongs with the
pipeline smoke tests, not here.
"""
from __future__ import annotations

import pytest

pytest.importorskip(
    "ultralytics",
    reason="ultralytics not installed in this environment (pre-existing gap; see README)",
)

from video_engine.contracts import ObjectClass  # noqa: E402
from video_engine.detection.yolo_detector import (  # noqa: E402
    BALL_AND_STUMPS_CLASSES,
    BALL_ONLY_CLASSES,
)


def test_ball_and_stumps_map_covers_both_classes():
    """The shape the pipeline was designed around, and the one calibration needs --
    stumps are the reference that turns pixels into metres on the pitch."""
    assert set(BALL_AND_STUMPS_CLASSES.values()) == {ObjectClass.BALL, ObjectClass.STUMPS}
    assert BALL_AND_STUMPS_CLASSES[0] is ObjectClass.BALL
    assert BALL_AND_STUMPS_CLASSES[1] is ObjectClass.STUMPS


def test_ball_only_map_declares_no_stumps_class():
    """The whole point of naming this map separately.

    A single-class cricket-ball checkpoint's class 0 *is* the ball, so it would appear to
    work under the two-class map -- right up until something asked for stumps. Declaring
    the absence is what lets `emits(ObjectClass.STUMPS)` answer honestly.
    """
    assert set(BALL_ONLY_CLASSES.values()) == {ObjectClass.BALL}
    assert ObjectClass.STUMPS not in BALL_ONLY_CLASSES.values()


def test_the_two_maps_agree_about_the_ball_index():
    """Swapping maps must change only which classes are declared, never what index 0
    means -- otherwise the same checkpoint would produce different detections under the
    two maps, which is a far worse trap than the one this is guarding."""
    assert BALL_ONLY_CLASSES[0] is BALL_AND_STUMPS_CLASSES[0]


def test_no_checkpoint_path_is_baked_into_the_module():
    """No URL or vendored path to anyone's weights. `ball_stumps_weights` is supplied by
    the operator, who is responsible for having the right to use it -- see the README.
    """
    import inspect

    from video_engine.detection import yolo_detector

    source = inspect.getsource(yolo_detector)
    assert "http://" not in source and "https://" not in source
    assert ".pt" not in source.replace('"yolov8n.pt"', "")  # the COCO default only
