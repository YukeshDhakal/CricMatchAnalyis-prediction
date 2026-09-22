"""Tests the decode bound, and the HTTP gate that applies it before a job exists.

The numbers in the first test are the ones from the production incident, written down as
assertions so the regression is a test failure rather than another dead container:
3840x2160, 598 frames, 13.1 MiB on disk, 13.86 GiB decoded at source resolution.
"""
from __future__ import annotations

import importlib
import io

import pytest

from api import frame_budget
from api.frame_budget import FrameBudgetError, plan_decode

# The real clip: C:\Users\acer\Downloads\13960806_3840_2160_60fps.mp4, 13.7 MB, 4K/60fps.
INCIDENT_WIDTH, INCIDENT_HEIGHT, INCIDENT_FRAMES = 3840, 2160, 598
_MIB = 1024 * 1024


def test_the_clip_that_took_production_down_is_now_bounded():
    """The whole fix in one assertion.

    Decoded at source resolution this clip is 13.86 GiB held for the length of the job.
    The plan brings it under 2 GiB by resizing to the largest edge anything downstream
    actually consumes.
    """
    unbounded_gib = INCIDENT_WIDTH * INCIDENT_HEIGHT * 3 * INCIDENT_FRAMES / 1024**3
    assert round(unbounded_gib, 2) == 13.86  # the measured footprint, not an estimate

    plan = plan_decode(INCIDENT_WIDTH, INCIDENT_HEIGHT, INCIDENT_FRAMES)

    assert plan.downscaled is True
    assert plan.decode_width == 1333
    assert plan.total_bytes / 1024**3 < 2.0
    # And the saving is the point: an order of magnitude, from a ratio nothing about the
    # 13.1 MiB upload hinted at.
    assert unbounded_gib / (plan.total_bytes / 1024**3) > 8


def test_footage_inside_the_budget_is_untouched():
    """720p is below every consumer's internal resize, so it is passed through exactly as
    before and its analysis is unchanged. A fix that silently altered results for footage
    that already worked would be trading one bug for a quieter one."""
    plan = plan_decode(1280, 720, 250)

    assert plan.downscaled is False
    assert plan.scale == 1.0
    assert (plan.decode_width, plan.decode_height) == (1280, 720)
    assert plan.as_payload()["note"].endswith("source resolution.")


def test_the_edge_limit_is_the_largest_edge_any_consumer_uses():
    """1333 is not a round number; it is torchvision's `KeypointRCNN` transform
    `max_size`. If that ever changes, this bound should change with it -- pinning it here
    is how that stays visible."""
    from torchvision.models.detection import keypointrcnn_resnet50_fpn

    model = keypointrcnn_resnet50_fpn(weights=None)
    assert frame_budget.DEFAULT_MAX_EDGE == model.transform.max_size


def test_aspect_ratio_survives_the_downscale():
    plan = plan_decode(3840, 2160, 100)
    assert plan.decode_width / plan.decode_height == pytest.approx(3840 / 2160, rel=1e-3)


def test_a_portrait_clip_is_bounded_on_its_long_edge():
    """Phone footage is taller than it is wide; bounding width would leave it unbounded."""
    plan = plan_decode(1080, 1920, 100)
    assert plan.decode_height == 1333
    assert plan.decode_width < plan.decode_height


def test_too_many_frames_is_refused_rather_than_truncated():
    with pytest.raises(FrameBudgetError, match="frames"):
        plan_decode(1280, 720, frame_budget.DEFAULT_MAX_FRAMES + 1)


def test_exactly_the_frame_limit_is_accepted():
    plan = plan_decode(1280, 720, frame_budget.DEFAULT_MAX_FRAMES)
    assert plan.source_frames == frame_budget.DEFAULT_MAX_FRAMES


def test_an_unknown_frame_count_budgets_for_the_worst_case():
    """OpenCV reports 0 frames for some containers. That must not read as "an empty clip,
    costing nothing" -- the estimate assumes the maximum, and `load_frames` enforces the
    real cap while decoding either way."""
    unknown = plan_decode(1280, 720, 0)
    known_max = plan_decode(1280, 720, frame_budget.DEFAULT_MAX_FRAMES)
    assert unknown.total_bytes == known_max.total_bytes


def test_a_file_with_no_readable_frame_size_is_refused():
    with pytest.raises(FrameBudgetError, match="not a video"):
        plan_decode(0, 0, 0)


def test_the_limits_are_configurable_from_the_environment(monkeypatch):
    monkeypatch.setenv("THIRD_UMPIRE_MAX_FRAME_EDGE", "640")
    monkeypatch.setenv("THIRD_UMPIRE_MAX_FRAMES", "10")
    assert frame_budget.max_edge() == 640
    with pytest.raises(FrameBudgetError):
        plan_decode(1280, 720, 11)


# --- the HTTP gate --------------------------------------------------------------------
# Refusing over-budget footage at submit time is the half that matters operationally: the
# alternative is what happened, where the clip was accepted on its byte size and the
# problem surfaced fourteen seconds later as a dead container.


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from api import supabase_auth

    monkeypatch.setenv("THIRD_UMPIRE_API_KEY", "test-key")
    monkeypatch.setenv("THIRD_UMPIRE_JOB_EXECUTOR", "thread")
    supabase_auth.clear_cache()

    import api.server as server_module

    importlib.reload(server_module)
    try:
        yield server_module, TestClient(server_module.app)
    finally:
        server_module._executor.shutdown(wait=False)


def _upload():
    return {"file": ("clip.mp4", io.BytesIO(b"pretend footage"), "video/mp4")}


def test_an_over_long_clip_is_413_before_a_job_is_created(client, monkeypatch):
    server_module, http = client
    monkeypatch.setattr(
        server_module,
        "probe_video",
        lambda path: (1280, 720, 60.0, frame_budget.DEFAULT_MAX_FRAMES + 500),
    )

    res = http.post("/jobs", files=_upload(), headers={"X-API-Key": "test-key"})

    assert res.status_code == 413
    assert "frames" in res.json()["detail"]
    # Nothing was queued: the caller gets an answer now instead of a job id that dies.
    assert server_module._jobs == {}


def test_a_4k_clip_within_the_frame_limit_is_accepted(client, monkeypatch):
    """The incident clip itself must still be *analysable* -- the fix is to bound what it
    costs, not to refuse 4K. A change that made this a 413 would have "fixed" the crash by
    removing the feature."""
    server_module, http = client
    monkeypatch.setattr(
        server_module,
        "probe_video",
        lambda path: (INCIDENT_WIDTH, INCIDENT_HEIGHT, 60.0, INCIDENT_FRAMES),
    )

    res = http.post("/jobs", files=_upload(), headers={"X-API-Key": "test-key"})
    assert res.status_code == 200


def test_a_file_that_is_not_video_is_a_400_not_a_failed_job(client, monkeypatch):
    server_module, http = client

    def not_a_video(path):
        raise ValueError(f"Could not open video file: {path}")

    monkeypatch.setattr(server_module, "probe_video", not_a_video)

    res = http.post("/jobs", files=_upload(), headers={"X-API-Key": "test-key"})
    assert res.status_code == 400
    assert server_module._jobs == {}
