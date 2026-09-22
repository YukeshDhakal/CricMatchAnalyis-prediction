"""A real, minimal HTTP API in front of the CV pipeline -- the piece the Video Pipeline
Test tab in the Streamlit app, and the `/app/video` screen in the Next.js web app, don't
have: a way to trigger a run from outside this machine.

**Why jobs, not a synchronous response.** A single clip takes anywhere from tens of
seconds to several minutes on CPU (see this project's README/MISTAKES.md -- pose
estimation dominates). Almost every HTTP host (including the Vercel app this is meant to
serve) times out a synchronous request long before that finishes. So: POST starts a job
and returns immediately with a job id; the caller polls GET until it's done. This is the
same shape any real "submit a video, get results later" product uses.

**Why one worker.** `KeypointRcnnPoseEstimator`/`YoloDetector` are real PyTorch/Ultralytics
models. Running two analyses at once on a small CPU host would slow both down and risk
OOM rather than actually parallelising -- serializing through one worker is deliberate,
not a placeholder for "add more later" without also adding more CPU/RAM.

**Two video sources, one of them guarded.** A job takes either an uploaded `file` or a
`video_url` this server fetches itself. The second is the same convenience the Streamlit
console has had, but it is a materially different feature here: on a public host, "fetch
this URL" is a request to make the *server* issue a request to an address the caller
chose. `api.video_source` is the guard that makes it safe to offer, and the reasoning
behind each rule it enforces is in that module's docstring.

**Why `include_frames` is opt-in.** Everything in a job result is a handful of scalars
except the sample frames, whose size scales with the footage. They are worth several
hundred kilobytes per job and only one caller (the web app's demo role) displays them, so
they are attached on request rather than by default.

**Why authorization is required, not optional.** Unlike the read-only Supabase tables the
web app queries directly, POST /jobs *spends real compute* (and therefore real hosting
cost) on every call. Refusing to start with nothing configured, and rejecting every write
without credentials, is deliberate: an accidentally-public, unauthenticated "run my
expensive ML pipeline" endpoint is a real cost/abuse risk, not just a nice-to-have.

**Two ways to authorize, for two different callers.** A request may present *either* a
static `X-API-Key` (the original path: scripts, curl smoke tests, anything internal and
trusted) *or* an `Authorization: Bearer <Supabase access token>` from a signed-in end
user (`api.supabase_auth`). The second exists because the browser now uploads here
directly rather than through the Next.js app -- a Vercel Serverless Function rejects a
video-sized request body at the platform gateway before any handler runs, so the proxy
that used to hold the static key could not carry real footage and has been removed. See
`api.supabase_auth`'s docstring for that story and for what verification actually happens.

The browser must never hold the static key, so the two paths are not interchangeable in
what they may ask for: a Supabase-authenticated caller gets `include_frames` decided
*for* them, from the role this server looked up. Only the static-key path -- which is by
definition already fully trusted -- may set it itself.

Run: `uvicorn api.server:app --host 0.0.0.0 --port 8000` (from `src/`, or with `src` on
PYTHONPATH). See `Dockerfile` at the repo root for how this is actually hosted.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api import supabase_auth
from api.pipeline_runner import run_analysis
from api.supabase_auth import SupabaseAuthError, SupabaseCaller
from api.video_source import VideoUrlError, fetch_video_url

API_KEY = os.environ.get("THIRD_UMPIRE_API_KEY")
MAX_UPLOAD_BYTES = int(os.environ.get("THIRD_UMPIRE_MAX_UPLOAD_MB", "50")) * 1024 * 1024
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "THIRD_UMPIRE_ALLOWED_ORIGINS",
        "https://third-umpire-ruby.vercel.app,http://localhost:3000,http://localhost:3210,http://localhost:3211",
    ).split(",")
    if o.strip()
]

app = FastAPI(title="Third Umpire video pipeline API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# One worker: see module docstring. A process-local dict is enough for a single-instance
# deployment; it does NOT survive a restart or scale past one instance -- a real job
# queue (Redis/DB-backed) would be the next step if this ever needs either.
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, "Job"] = {}
_jobs_lock = threading.Lock()

JobStatus = Literal["queued", "running", "done", "error"]


class Job(BaseModel):
    id: str
    status: JobStatus
    created_at: float
    source_label: str
    result: dict[str, Any] | None = None
    error: str | None = None
    # Who may read this job back. Set for jobs started by a signed-in end user, None for
    # jobs started with the static key. Excluded from the response body: it is an access
    # control fact, not something a polling client needs, and echoing a user id to
    # whoever asks is the opposite of the point.
    owner_user_id: str | None = Field(default=None, exclude=True)


def _require_api_key(x_api_key: str | None) -> None:
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="THIRD_UMPIRE_API_KEY is not configured on this server -- refusing to "
            "run the pipeline rather than leave an unauthenticated, compute-spending "
            "endpoint open.",
        )
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


@dataclass(frozen=True)
class Caller:
    """Who is making this request, and therefore what they are allowed to ask for.

    `kind == "api_key"` is the trusted internal path (whoever holds the static key can
    already do anything this service does). `kind == "supabase"` is a real end user with
    a verified identity and a role read from the database, and carries `user`.
    """

    kind: Literal["api_key", "supabase"]
    user: SupabaseCaller | None = None


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return token.strip() or None


def _authorize(x_api_key: str | None, authorization: str | None) -> Caller:
    """Accept either credential, and say which one was used.

    Order matters, and it is: an `X-API-Key` header, once *present*, is validated
    strictly and nothing else is consulted. That keeps every existing scripted caller
    byte-identical -- a wrong key still gets the same 401 it always did, rather than
    silently falling through to some other check and producing a different error.
    """
    if x_api_key is not None:
        _require_api_key(x_api_key)
        return Caller(kind="api_key")

    token = _bearer_token(authorization)
    if token:
        try:
            return Caller(kind="supabase", user=supabase_auth.resolve_caller(token))
        except SupabaseAuthError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    # No credential at all. If bearer auth isn't even configured here, the original
    # behaviour is still the right one (503 when the server holds no key, 401 otherwise);
    # if it is configured, say so, because "send an X-API-Key" is the wrong instruction
    # for a browser that can never have one.
    if not supabase_auth.is_configured():
        _require_api_key(x_api_key)
    raise HTTPException(
        status_code=401,
        detail="Missing credentials. Send either an X-API-Key header or an "
        "'Authorization: Bearer <Supabase access token>' header.",
    )


def _run_job(
    job_id: str,
    video_path: Path,
    source_label: str,
    match_id,
    innings,
    over,
    ball,
    include_frames: bool = False,
) -> None:
    with _jobs_lock:
        _jobs[job_id].status = "running"
    try:
        result = run_analysis(
            video_path, source_label, match_id, innings, over, ball, include_frames
        )
        with _jobs_lock:
            _jobs[job_id].status = "done"
            _jobs[job_id].result = result
    except Exception as exc:  # noqa: BLE001 -- a failed job is a valid, reportable outcome
        with _jobs_lock:
            _jobs[job_id].status = "error"
            _jobs[job_id].error = str(exc)
    finally:
        video_path.unlink(missing_ok=True)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness only -- deliberately does not touch the pipeline/models, so a load
    balancer's health check never pays model-load cost or blocks on the worker queue."""
    return {"status": "ok"}


@app.post("/jobs")
async def create_job(
    file: UploadFile | None = File(None),
    video_url: str | None = Form(None),
    match_id: str | None = Form(None),
    innings: int | None = Form(None),
    over: int | None = Form(None),
    ball: int | None = Form(None),
    include_frames: bool = Form(False),
    x_api_key: str | None = Header(None),
    authorization: str | None = Header(None),
) -> dict[str, str]:
    """Starts an analysis job. Supply exactly one video source: an uploaded `file`, or a
    `video_url` this server fetches itself (see `api.video_source` for what it will and
    won't fetch, and why that guard is not optional on a public host).

    Authorize with either an `X-API-Key` header or `Authorization: Bearer <Supabase
    access token>`; see the module docstring for why there are two.

    `include_frames` opts in to the expensive part of the response -- the per-track table
    and a few real frames with detections and pose keypoints drawn on them. It defaults to
    false so the ordinary response stays small; see `pipeline_runner.run_analysis`.
    **A Supabase-authenticated caller does not get to set it**: whatever the form field
    says is discarded and the value is re-derived from the role this server looked up.
    A browser asking for `include_frames=true` is asking to spend more of someone else's
    CPU, so the request is treated as a suggestion from an untrusted party -- which is
    exactly what it is.
    """
    caller = await run_in_threadpool(_authorize, x_api_key, authorization)

    # The one line the whole role system exists for. Note that it *overwrites* rather
    # than reads-if-absent: a forged `include_frames=true` in the form must not be able
    # to switch on the expensive path, and the only way to guarantee that is to never
    # consult the client's value on this path at all.
    if caller.kind == "supabase":
        assert caller.user is not None
        include_frames = supabase_auth.wants_frames(caller.user.role)

    has_file = file is not None and bool(file.filename)
    if has_file == bool(video_url):
        raise HTTPException(
            status_code=400,
            detail="Supply exactly one of 'file' or 'video_url'."
            if not has_file and not video_url
            else "Supply either 'file' or 'video_url', not both.",
        )

    if has_file:
        assert file is not None  # narrowed by has_file; keeps type checkers honest
        suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
        fd, tmp_path_str = tempfile.mkstemp(suffix=suffix, prefix="third_umpire_")
        tmp_path = Path(tmp_path_str)
        size = 0
        with os.fdopen(fd, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    out.close()
                    tmp_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
                    )
                out.write(chunk)
        source_label = file.filename or tmp_path.name
    else:
        assert video_url is not None
        suffix = Path(urlparse(video_url).path).suffix or ".mp4"
        fd, tmp_path_str = tempfile.mkstemp(suffix=suffix, prefix="third_umpire_")
        os.close(fd)  # fetch_video_url opens the path itself
        tmp_path = Path(tmp_path_str)
        try:
            # Deliberately synchronous, before the job is created. A URL that can't be
            # fetched is the caller's mistake and is worth a 400 they see immediately,
            # rather than a job id that fails a minute later for a reason they then have
            # to poll for.
            await run_in_threadpool(fetch_video_url, video_url, tmp_path, MAX_UPLOAD_BYTES)
        except VideoUrlError as exc:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        source_label = Path(urlparse(video_url).path).name or video_url

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = Job(
            id=job_id,
            status="queued",
            created_at=time.time(),
            source_label=source_label,
            owner_user_id=caller.user.user_id if caller.user else None,
        )

    _executor.submit(
        _run_job, job_id, tmp_path, source_label, match_id, innings, over, ball, include_frames
    )
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, x_api_key: str | None = Header(None), authorization: str | None = Header(None)) -> Job:
    """Reads a job back. Authorized the same two ways as POST, minus the role check --
    any granted account may poll.

    Reading spends no pipeline compute, and a job id is an unguessable UUID, so this was
    open for a while. It is closed now for a narrower reason: a finished job's result can
    carry sampled frames from the submitted clip, so a leaked id would hand over the
    footage's contents and not just a status word. That gate used to sit in the Next.js
    proxy; the proxy is gone, so it sits here.

    A signed-in user may read only their *own* jobs -- not every job on the server. With
    two accounts and unguessable ids this is a narrow gap, but it costs one comparison
    and it is the difference between "the demo account can see its own footage" and "the
    demo account can see any footage anyone uploaded". Someone else's job answers 404
    rather than 403, so the response says nothing about whether that id exists. The
    static-key path is unrestricted, because it always was and whoever holds that key is
    already fully trusted.
    """
    caller = _authorize(x_api_key, authorization)

    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job.")
    if caller.kind == "supabase":
        assert caller.user is not None
        if job.owner_user_id != caller.user.user_id:
            raise HTTPException(status_code=404, detail="No such job.")
    return job
