"""Writing a finished analysis into Supabase's `video_analyses`, owned by whoever ran it.

**Why this exists at all.** Until now a job's result went into the HTTP response and
nowhere else. Every run a user started was gone the moment they closed the tab: the
`video_analyses` table held exactly two rows, both exported from the professional-player
demo data by `scripts/export_for_lovable.py`, and nothing in this service had ever written
to it. A product whose value is "your footage, analysed, over time" cannot have an
ephemeral result as its only output.

**Why the caller's own token, and not a service-role key.** The insert is sent with the
same bearer token the caller authenticated with, so PostgREST evaluates the table's insert
policy (`with check (user_id = auth.uid())`) as that user. Two things follow, and both are
the reason this is shaped this way:

- Ownership cannot be forged, by this service or through it. The row's `user_id` has to
  equal the token's subject or Postgres refuses the write. This service does not get to be
  trusted about whose data something is; it is not in a position to be wrong about it.
- No new secret. A service-role key would bypass RLS entirely, would have to be stored on
  Railway, and would make this container a far more interesting thing to compromise. The
  same reasoning as `api.supabase_auth`: `SUPABASE_URL` and `SUPABASE_ANON_KEY` are the two
  public values the web app already ships to every browser.

**Why failure here is not job failure.** The analysis is the expensive part and it
succeeded; the result is real and is about to be returned. Throwing it away because a
database write failed would be destroying minutes of real compute over a retryable
problem. So this returns an outcome instead of raising, the job still completes, and the
response carries `persisted: false` with the reason -- which the web app renders as a
notice on the result, because a run that is on screen but not in anyone's history is a run
that vanishes on reload, and finding that out by reloading is the worse way to learn it.

**Only these columns.** A job result carries several keys that `video_analyses` has no
column for -- `timings`, `frame_budget`, `pitch_geometry`, and (for the demo role)
`tracks` and `sample_frames`, the last of which is base64 frame data measured in hundreds
of kilobytes. PostgREST rejects an insert naming a column that does not exist, so the
payload is built from an explicit allow-list rather than by handing over the result dict
and hoping. The allow-list is also the honest description of what persists: the history
below the live panel is the scalar summary, not the frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

_TIMEOUT_SECONDS = 15

# Exactly the columns `public.video_analyses` has, minus `id` and `analyzed_at` (both
# defaulted by the database) and plus `user_id`, which is set from the verified token
# rather than from anything in the result.
_PERSISTED_KEYS = (
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
)


@dataclass(frozen=True)
class PersistOutcome:
    """What happened, in a form the caller can attach to the job result verbatim."""

    persisted: bool
    error: str | None = None


def row_for(result: dict[str, Any], user_id: str) -> dict[str, Any]:
    """Project a pipeline result onto the table's columns, owned by `user_id`."""
    row: dict[str, Any] = {key: result[key] for key in _PERSISTED_KEYS if key in result}
    row["user_id"] = user_id
    return row


def persist_analysis(
    *,
    supabase_url: str,
    anon_key: str,
    token: str,
    user_id: str,
    result: dict[str, Any],
) -> PersistOutcome:
    """Insert one completed analysis as the token's own user. Never raises."""
    try:
        res = requests.post(
            f"{supabase_url.rstrip('/')}/rest/v1/video_analyses",
            json=row_for(result, user_id),
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": anon_key,
                "Content-Type": "application/json",
                # Nothing here reads the inserted row back, and asking PostgREST to return
                # it would mean the insert also needs a SELECT policy match to succeed --
                # coupling a write to a read for a value that gets discarded.
                "Prefer": "return=minimal",
            },
            timeout=_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return PersistOutcome(False, f"Couldn't reach Supabase to save this run: {exc}")

    if res.status_code in (200, 201, 204):
        return PersistOutcome(True)

    # 401/403 here is the interesting one: it means the token was good enough to start the
    # job and is no longer good enough to write (expired during a long run), or the insert
    # policy is not what this code assumes. Either way the caller should see it rather than
    # get a silent "your run wasn't saved".
    detail = ""
    try:
        body = res.json()
        if isinstance(body, dict):
            detail = body.get("message") or body.get("hint") or ""
    except ValueError:
        detail = ""
    suffix = f" ({detail})" if detail else ""
    return PersistOutcome(False, f"Supabase refused to save this run: HTTP {res.status_code}{suffix}")
