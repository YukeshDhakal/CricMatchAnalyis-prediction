"""Tests `api.analysis_store` -- the one place this service writes to the database.

The HTTP is stubbed, for the same reason it is in test_supabase_auth.py: a unit suite that
needed live Supabase credentials would be unrunnable in CI, and one that wrote for real
would put test rows in the live project. What is stubbed is the transport. The payload
construction, the header set, the status mapping and above all the never-raises contract
are the real code -- and that contract is the load-bearing one here, because this runs on
a completion callback where an exception would mean a finished analysis is never recorded
as finished at all.
"""
from __future__ import annotations

import pytest
import requests

from api import analysis_store

USER_ID = "5a3f52b0-d3ad-4756-99bc-3d99c2e0140d"

RESULT = {
    "source_label": "net-session.mp4",
    "match_id": "m1",
    "innings": 1,
    "over_number": 2,
    "ball_number": 3,
    "frames_processed": 120,
    "people_tracked": 2,
    "pose_frames": 96,
    "shot_classification": "unknown",
    "event_notes": "no ball track",
    "detection_confidence": {"player": {"count": 12, "mean": 0.71}},
    # Everything below here is in a real result and has no column in video_analyses.
    "timings": {"detection": 1.5},
    "frame_budget": {"decode_width": 1280},
    "pitch_geometry": {"available": False},
    "tracks": [{"track_id": 1}],
    "sample_frames": [{"image_base64": "x" * 5000}],
}


class FakeResponse:
    def __init__(self, status_code, payload=None, *, raw=False):
        self.status_code = status_code
        self._payload = payload
        self._raw = raw

    def json(self):
        if self._raw:
            raise ValueError("not json")
        return self._payload


class FakePost:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": dict(headers or {})})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _install(monkeypatch, response):
    fake = FakePost(response)
    monkeypatch.setattr(analysis_store.requests, "post", fake)
    return fake


def _persist(**overrides):
    kwargs = {
        "supabase_url": "https://project.supabase.co",
        "anon_key": "anon-key",
        "token": "user-token",
        "user_id": USER_ID,
        "result": RESULT,
    }
    kwargs.update(overrides)
    return analysis_store.persist_analysis(**kwargs)


# --- what gets sent --------------------------------------------------------------------


def test_only_real_columns_are_sent():
    """PostgREST rejects an insert naming a column that does not exist, so sending the
    result dict wholesale would turn every completed run into an unsaved one. The frames
    are the other half of the point: several hundred kilobytes with nowhere to go."""
    row = analysis_store.row_for(RESULT, USER_ID)
    assert set(row) == {
        "user_id",
        "source_label",
        "match_id",
        "innings",
        "over_number",
        "ball_number",
        "frames_processed",
        "people_tracked",
        "pose_frames",
        "shot_classification",
        "event_notes",
        "detection_confidence",
    }
    assert row["detection_confidence"] == {"player": {"count": 12, "mean": 0.71}}


def test_the_owner_comes_from_the_argument_not_the_result():
    """The result dict is produced by the pipeline and could in principle be made to carry
    anything. Ownership comes from the verified token's user id, passed in separately --
    and RLS checks it again on the way in regardless."""
    tampered = dict(RESULT, user_id="somebody-else")
    assert analysis_store.row_for(tampered, USER_ID)["user_id"] == USER_ID


def test_a_partial_result_sends_only_the_keys_it_has():
    """A result missing an optional key must not become a literal null for a NOT NULL
    column, or an insert of the string "None"."""
    row = analysis_store.row_for({"source_label": "clip.mp4"}, USER_ID)
    assert row == {"source_label": "clip.mp4", "user_id": USER_ID}


def test_the_insert_is_sent_as_the_user(monkeypatch):
    fake = _install(monkeypatch, FakeResponse(201))
    assert _persist().persisted is True

    call = fake.calls[0]
    assert call["url"] == "https://project.supabase.co/rest/v1/video_analyses"
    # The bearer is the *user's*, not a service-role key -- that is what makes RLS, rather
    # than this service's good intentions, the thing that decides the row's owner.
    assert call["headers"]["Authorization"] == "Bearer user-token"
    assert call["headers"]["apikey"] == "anon-key"
    assert call["headers"]["Prefer"] == "return=minimal"
    assert call["json"]["user_id"] == USER_ID


def test_a_trailing_slash_on_the_url_does_not_double_up(monkeypatch):
    fake = _install(monkeypatch, FakeResponse(201))
    _persist(supabase_url="https://project.supabase.co/")
    assert fake.calls[0]["url"] == "https://project.supabase.co/rest/v1/video_analyses"


# --- what comes back -------------------------------------------------------------------


@pytest.mark.parametrize("status", [200, 201, 204])
def test_every_success_status_postgrest_uses_counts_as_saved(monkeypatch, status):
    """`Prefer: return=minimal` makes 201 a 204. Both are success and neither should be
    reported to a user as a failed save."""
    _install(monkeypatch, FakeResponse(status))
    assert _persist().persisted is True


def test_a_refusal_is_reported_and_not_raised(monkeypatch):
    """401/403 here means the token expired during a long run, or the insert policy is not
    what this code assumes. Either way the caller has to be told, and the job must still
    complete."""
    _install(monkeypatch, FakeResponse(401, {"message": "JWT expired"}))
    outcome = _persist()
    assert outcome.persisted is False
    assert "401" in outcome.error
    assert "JWT expired" in outcome.error


def test_a_refusal_with_an_unreadable_body_still_produces_a_message(monkeypatch):
    _install(monkeypatch, FakeResponse(502, raw=True))
    outcome = _persist()
    assert outcome.persisted is False
    assert "502" in outcome.error


def test_an_unreachable_supabase_is_reported_and_not_raised(monkeypatch):
    """The contract that matters most. This runs on a job-completion callback: an
    exception escaping here would leave a finished analysis never marked finished, turning
    a database blip into a job that polls forever."""
    _install(monkeypatch, requests.ConnectionError("dns exploded"))
    outcome = _persist()
    assert outcome.persisted is False
    assert "dns exploded" in outcome.error


def test_a_timeout_is_reported_and_not_raised(monkeypatch):
    _install(monkeypatch, requests.Timeout("took too long"))
    assert _persist().persisted is False


def test_the_token_is_not_in_the_error_message(monkeypatch):
    """Errors from here go into a job result the browser renders. A credential must not
    ride along into the page."""
    _install(monkeypatch, requests.ConnectionError("failed talking to host"))
    assert "user-token" not in (_persist().error or "")
