"""Tests for the classical motion-energy signal.

Synthetic frames, and deliberately so: the property under test is "does a moving patch
score above a still one", which is exactly reproducible in constructed pixels. The real
footage evidence that motivated this module (stationary balls detected at 0.44-0.55 for
83 frames) is recorded in `test_trajectory_real_footage.py` and the module docstring --
this file checks the mechanism, not the claim about the checkpoint.
"""
import numpy as np

from video_engine.contracts import BoundingBox, Detection, ObjectClass
from video_engine.motion import (
    STATIONARY_ENERGY_CEILING,
    global_motion_ratio,
    motion_energy,
    score_detections,
)


def _frame(value: int = 0, size: int = 80) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _with_square(value: int, cx: int, cy: int, half: int = 6, size: int = 80) -> np.ndarray:
    frame = _frame(30, size)
    frame[cy - half : cy + half, cx - half : cx + half] = value
    return frame


def _box(cx: float, cy: float, half: float = 6.0) -> BoundingBox:
    return BoundingBox(cx - half, cy - half, cx + half, cy + half)


def test_a_moving_patch_scores_high():
    frames = [_with_square(220, 20, 40), _with_square(220, 40, 40), _with_square(220, 60, 40)]

    assert motion_energy(frames, 1, _box(40, 40)) > STATIONARY_ENERGY_CEILING


def test_a_stationary_patch_scores_zero():
    frames = [_with_square(220, 40, 40) for _ in range(3)]

    assert motion_energy(frames, 1, _box(40, 40)) == 0.0


def test_clip_boundaries_report_no_energy_rather_than_a_one_sided_guess():
    frames = [_with_square(220, 20, 40), _with_square(220, 40, 40), _with_square(220, 60, 40)]

    assert motion_energy(frames, 0, _box(20, 40)) == 0.0
    assert motion_energy(frames, 2, _box(60, 40)) == 0.0


def test_a_box_outside_the_frame_scores_zero_instead_of_raising():
    frames = [_frame() for _ in range(3)]

    assert motion_energy(frames, 1, BoundingBox(-50, -50, -10, -10)) == 0.0


def test_score_detections_flags_moving_and_still_detections_apart():
    frames = [_with_square(220, 20, 40), _with_square(220, 45, 40), _with_square(220, 70, 40)]
    # A second, genuinely static object that the detector also fires on.
    for f in frames:
        f[10:22, 10:22] = 200

    moving = Detection(1, ObjectClass.BALL, _box(45, 40), 0.4)
    resting = Detection(1, ObjectClass.BALL, _box(16, 16), 0.9)

    evidence = score_detections(frames, [moving, resting])

    assert evidence[0].is_moving is True
    assert evidence[1].is_moving is False
    assert evidence[0].energy > evidence[1].energy


def test_global_motion_ratio_separates_a_fixed_camera_from_a_pan():
    still = [_with_square(220, 40, 40) for _ in range(3)]
    small_move = [_with_square(220, 38, 40), _with_square(220, 42, 40), _with_square(220, 46, 40)]
    # A pan only registers against texture -- a flat background shifts into itself and
    # looks static, which is a property of frame differencing worth stating in a test
    # rather than discovering on footage. The stripes are high-contrast on purpose: a
    # gentle gradient shifted a few pixels changes each one by less than the threshold.
    stripes = np.tile(
        (np.arange(80) % 2 * 200).astype(np.uint8)[None, :, None], (80, 1, 3)
    )
    panning = [np.roll(stripes, shift, axis=1) for shift in range(3)]

    assert global_motion_ratio(still, 1) == 0.0
    assert global_motion_ratio(small_move, 1) < 0.2
    assert global_motion_ratio(panning, 1) > 0.5
