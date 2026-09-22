"""Tests the HTTP layer (auth, job lifecycle, error handling) without paying for real
model loads or inference -- `run_analysis` is monkeypatched to a fast fake. Whether the
*pipeline itself* produces correct results is covered by the video_engine test suite;
this file is about whether the API around it behaves (rejects unauthenticated writes,
returns a job id, reaches "done", 404s on an unknown id), which is a different, real
risk from the pipeline's own correctness.
"""
from __future__ import annotations

import importlib
import io
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("THIRD_UMPIRE_API_KEY", "test-key")
    monkeypatch.setenv("THIRD_UMPIRE_MAX_UPLOAD_MB", "1")

    import api.server as server_module

    importlib.reload(server_module)  # picks up the monkeypatched env vars at import time

    def fake_run_analysis(
        video_path, source_label, match_id, innings, over, ball, include_frames=False
    ):
        result = {
            "source_label": source_label,
            "match_id": match_id,
            "innings": innings,
            "over_number": over,
            "ball_number": ball,
            "frames_processed": 1,
            "people_tracked": 0,
            "pose_frames": 0,
            "shot_classification": "unknown",
            "event_notes": "fake result for a test",
            "detection_confidence": {},
            "timings": {},
        }
        # The fake records what it was *asked* for rather than producing real frames --
        # whether the frames themselves are real pictures is tests/video_engine/
        # test_overlay.py's job. What this file has to prove is that the flag survives
        # the HTTP boundary, because a silently-dropped `include_frames` would look
        # exactly like "the model found nothing to draw".
        result["include_frames_requested"] = include_frames
        if include_frames:
            result["tracks"] = []
            result["sample_frames"] = []
        return result

    monkeypatch.setattr(server_module, "run_analysis", fake_run_analysis)
    try:
        yield TestClient(server_module.app)
    finally:
        server_module._executor.shutdown(wait=False)


def _tiny_upload():
    return {"file": ("clip.mp4", io.BytesIO(b"not a real video, just test bytes"), "video/mp4")}


def test_health_needs_no_key(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_create_job_without_api_key_is_rejected(client):
    res = client.post("/jobs", files=_tiny_upload())
    assert res.status_code == 401


def test_create_job_with_wrong_api_key_is_rejected(client):
    res = client.post("/jobs", files=_tiny_upload(), headers={"X-API-Key": "wrong"})
    assert res.status_code == 401


def test_create_job_and_poll_to_completion(client):
    res = client.post(
        "/jobs",
        files=_tiny_upload(),
        data={"match_id": "m1", "innings": "1", "over": "2", "ball": "3"},
        headers={"X-API-Key": "test-key"},
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    assert res.json()["status"] == "queued"

    # The fake runs synchronously fast, but it's still on a background thread -- poll
    # briefly rather than assuming it's already done the instant the response returns.
    deadline = time.time() + 5
    job = None
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.05)

    assert job is not None
    assert job["status"] == "done", job
    assert job["result"]["shot_classification"] == "unknown"
    assert job["result"]["match_id"] == "m1"
    assert job["result"]["over_number"] == 2


def test_unknown_job_id_is_404(client):
    res = client.get("/jobs/does-not-exist")
    assert res.status_code == 404


def test_oversized_upload_is_rejected(client):
    big = io.BytesIO(b"x" * (2 * 1024 * 1024))  # 2MB against the fixture's 1MB limit
    res = client.post(
        "/jobs", files={"file": ("clip.mp4", big, "video/mp4")}, headers={"X-API-Key": "test-key"}
    )
    assert res.status_code == 413


def _await_job(client, job_id, timeout=5):
    deadline = time.time() + timeout
    job = None
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    return job


# --- include_frames -------------------------------------------------------------------
# The expensive half of the response. It is opt-in, so the two things worth asserting are
# that it stays off unless asked for, and that asking for it actually reaches the runner.


def test_frames_are_not_included_by_default(client):
    res = client.post("/jobs", files=_tiny_upload(), headers={"X-API-Key": "test-key"})
    job = _await_job(client, res.json()["job_id"])
    assert job["status"] == "done", job
    assert job["result"]["include_frames_requested"] is False
    assert "sample_frames" not in job["result"]
    assert "tracks" not in job["result"]


@pytest.mark.parametrize("sent", ["true", "True", "1"])
def test_include_frames_reaches_the_runner(client, sent):
    """Form fields are strings on the wire; FastAPI coerces them to the declared bool.
    Parametrised because a client sending "1" instead of "true" silently getting the
    small response would be a confusing bug to chase from the frontend."""
    res = client.post(
        "/jobs",
        files=_tiny_upload(),
        data={"include_frames": sent},
        headers={"X-API-Key": "test-key"},
    )
    job = _await_job(client, res.json()["job_id"])
    assert job["status"] == "done", job
    assert job["result"]["include_frames_requested"] is True
    assert job["result"]["sample_frames"] == []
    assert job["result"]["tracks"] == []


def test_include_frames_false_is_respected(client):
    res = client.post(
        "/jobs",
        files=_tiny_upload(),
        data={"include_frames": "false"},
        headers={"X-API-Key": "test-key"},
    )
    job = _await_job(client, res.json()["job_id"])
    assert job["result"]["include_frames_requested"] is False


# --- video_url ------------------------------------------------------------------------


def test_no_video_source_at_all_is_a_400(client):
    res = client.post("/jobs", data={"match_id": "m1"}, headers={"X-API-Key": "test-key"})
    assert res.status_code == 400
    assert "exactly one" in res.json()["detail"]


def test_both_sources_at_once_is_a_400(client):
    res = client.post(
        "/jobs",
        files=_tiny_upload(),
        data={"video_url": "https://example.com/clip.mp4"},
        headers={"X-API-Key": "test-key"},
    )
    assert res.status_code == 400
    assert "not both" in res.json()["detail"]


def test_a_rejected_video_url_is_a_400_with_the_reason_not_a_queued_job(client):
    """A URL the server won't fetch is the caller's mistake. They should see it on the
    POST, not discover it by polling a job that fails a minute later."""
    res = client.post(
        "/jobs",
        data={"video_url": "https://www.youtube.com/watch?v=abc"},
        headers={"X-API-Key": "test-key"},
    )
    assert res.status_code == 400
    assert "direct" in res.json()["detail"]
    assert "job_id" not in res.json()


def test_video_url_still_needs_the_api_key(client):
    res = client.post("/jobs", data={"video_url": "https://example.com/clip.mp4"})
    assert res.status_code == 401


def test_a_fetched_video_url_becomes_a_job(client, monkeypatch, tmp_path):
    """The fetch itself is covered by tests/api/test_video_source.py against a real local
    server; here it's stubbed so this test stays about the HTTP layer -- does a good URL
    produce a job whose source_label is the file's name rather than the whole URL."""
    import api.server as server_module

    def fake_fetch(url, dest_path, max_bytes):
        dest_path.write_bytes(b"pretend video bytes")
        return dest_path

    monkeypatch.setattr(server_module, "fetch_video_url", fake_fetch)

    res = client.post(
        "/jobs",
        data={"video_url": "https://cdn.example.com/clips/over3.mp4", "include_frames": "true"},
        headers={"X-API-Key": "test-key"},
    )
    assert res.status_code == 200, res.text
    job = _await_job(client, res.json()["job_id"])
    assert job["status"] == "done", job
    assert job["source_label"] == "over3.mp4"
    assert job["result"]["include_frames_requested"] is True
