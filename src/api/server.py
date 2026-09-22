"""A real, minimal HTTP API in front of the CV pipeline -- the piece the Video Pipeline
Test tab in the Streamlit app, and the `/app/video` screen in the Next.js web app, don't
have: a way to trigger a run from outside this machine.

**Why jobs, not a synchronous response.** A single clip takes anywhere from tens of
seconds to several minutes on CPU (see this project's README/MISTAKES.md -- pose
estimation dominates). Almost every HTTP host (including the Vercel app this is meant to
serve) times out a synchronous request long before that finishes. So: POST starts a job
and returns immediately with a job id; the caller polls GET until it's done. This is the
same shape any real "submit a video, get results later" product uses.

**Why one worker, and why it is a process.** `KeypointRcnnPoseEstimator`/`YoloDetector`
are real PyTorch/Ultralytics models. Running two analyses at once on a small CPU host
would slow both down and risk OOM rather than actually parallelising -- serializing
through one worker is deliberate, not a placeholder for "add more later" without also
adding more CPU/RAM. That worker used to be a thread and is now a child process, so a
crash or an OOM kill in the decode/inference stack costs one job instead of the whole
service; `api.job_executor` has the reasoning and the measurement behind it.

**Why a byte-size limit is not enough.** `THIRD_UMPIRE_MAX_UPLOAD_MB` bounds what crosses
the wire, which turns out to bound almost nothing that matters: the 13.1 MiB 4K clip that
took production down decoded to 13.86 GiB of frames. Every job is now also checked
against `api.frame_budget` -- in decoded pixels, before the job is created -- and the
decoder enforces the same bound while it reads. That module is where the numbers and the
consequences are written down.

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
from collections import deque
from concurrent.futures import CancelledError, Future
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api import frame_budget, supabase_auth
from api.frame_budget import FrameBudgetError
from api.job_executor import JobExecutor, run_pipeline_job
from api.pipeline_runner import probe_video
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


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Let the worker process go when the server stops.

    `wait=False`: a shutdown that blocks until a multi-minute analysis finishes is a
    shutdown the platform SIGKILLs instead, which is a worse way to end. Without this the
    spawned worker can outlive its parent and keep a CPU busy with a job nobody can
    collect -- a thread pool had no such problem, a process pool does.
    """
    yield
    _executor.shutdown(wait=False)


app = FastAPI(title="Third Umpire video pipeline API", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# One worker, in its own OS process: see `api.job_executor` for why a process and not a
# thread, and for the measurement that ruled out the event-loop-starvation theory this
# was first blamed on. A process-local dict is enough for a single-instance deployment;
# it does NOT survive a restart or scale past one instance -- a real job queue
# (Redis/DB-backed) would be the next step if this ever needs either.
_executor = JobExecutor()
_jobs: dict[str, "Job"] = {}
# The one worker takes jobs in submission order, so the head of this queue is the job it
# is on and the rest are waiting. Kept in the parent because that is where the HTTP
# handlers read it; the worker never sees job state at all. This is what replaces the old
# arrangement where the worker itself flipped a job to "running" -- it cannot any more,
# it is in a different process.
_pending: deque[str] = deque()
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


def _finish_job(job_id: str, status: JobStatus, *, result=None, error=None) -> None:
    """Record a terminal outcome and hand the worker to the next job in line."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.status = status
            job.result = result
            job.error = error
        try:
            _pending.remove(job_id)
        except ValueError:  # already advanced past, or never queued
            pass
        if _pending:
            head = _jobs.get(_pending[0])
            if head is not None and head.status == "queued":
                head.status = "running"


def _on_job_done(job_id: str, video_path: Path, future: Future) -> None:
    """Runs in the parent process when the worker's future resolves.

    Deleting the upload happens **here**, not in the worker. The worker's `finally` used
    to do it, which works right up until the worker is the thing that dies -- a process
    killed by a segfault or by the OOM killer runs no `finally`, and the temp file would
    leak on exactly the failures most likely to repeat.
    """
    try:
        exc = future.exception()
    except CancelledError:
        _finish_job(job_id, "error", error="Job was cancelled before it ran.")
        video_path.unlink(missing_ok=True)
        return

    if exc is not None:
        if _executor.discard_if_broken(exc):
            # The worker process died rather than returning a failure: a native crash in
            # the decode/inference stack, or the kernel killing it. Worth distinguishing
            # from a pipeline error, because "this clip broke the analyser" and "the
            # analyser had a bad day" call for different responses from whoever reads it.
            #
            # Note this reaches *every* outstanding job, not just the one that was
            # running: a broken pool fails its whole queue, and a job that never got to
            # run is not being re-submitted automatically here. If the clip that killed
            # the worker is the one being retried, an automatic retry is an infinite
            # loop; telling the caller plainly and letting them decide is the honest
            # option and the one that terminates.
            _finish_job(
                job_id,
                "error",
                error=(
                    "The analysis worker process died (killed or crashed rather than "
                    "failing normally), so this job did not complete. The service is "
                    "still up and a fresh worker will take the next job -- resubmit to "
                    "retry."
                ),
            )
        else:
            _finish_job(job_id, "error", error=f"{type(exc).__name__}: {exc}")
    else:
        outcome = future.result()
        if outcome.get("ok"):
            _finish_job(job_id, "done", result=outcome["result"])
        else:
            _finish_job(job_id, "error", error=outcome.get("error", "Unknown pipeline error."))

    video_path.unlink(missing_ok=True)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness only -- deliberately does not touch the pipeline/models, so a load
    balancer's health check never pays model-load cost or blocks on the worker queue.

    It is also the endpoint that stays answerable while a job runs, which is now a
    structural guarantee rather than a hope: the pipeline executes in a different process
    (`api.job_executor`), so no amount of CPU or memory pressure inside it can occupy the
    thread serving this. `tests/api/test_server_responsiveness.py` holds that property
    down. Worth pointing a Railway `healthcheckPath` at -- there is none configured
    today, which is why the platform had no app-level signal during the incident."""
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

    # Decided here, before a job exists, because the alternative is what actually
    # happened in production: a clip is accepted on its byte size, the job starts, and
    # fourteen seconds later the decoder has produced 13.86 GiB of frames and the kernel
    # kills the container. `MAX_UPLOAD_BYTES` cannot see any of that -- the ratio between
    # an H.264 file and its decoded frames is a property of the codec, and for 4K it is
    # about 1000x. `api.frame_budget` carries the measurements.
    try:
        width, height, _fps, frame_count = await run_in_threadpool(probe_video, tmp_path)
        frame_budget.plan_decode(width, height, frame_count)
    except FrameBudgetError as exc:
        tmp_path.unlink(missing_ok=True)
        # 413 rather than 400: this is "too large for me to process", the same class of
        # answer as the upload-size limit, just measured in the unit that matters.
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:  # probe_video: not a video, or an undecodable codec
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = Job(
            id=job_id,
            status="queued",
            created_at=time.time(),
            source_label=source_label,
            owner_user_id=caller.user.user_id if caller.user else None,
        )
        _pending.append(job_id)
        # "running" means "the single worker is on this one", which with one worker is
        # exactly "nothing is ahead of it". The worker used to set this itself; it is in
        # another process now and deliberately knows nothing about job state.
        if len(_pending) == 1:
            _jobs[job_id].status = "running"

    try:
        future = _executor.submit(
            run_pipeline_job,
            tmp_path,
            source_label,
            match_id,
            innings,
            over,
            ball,
            include_frames,
        )
    except Exception as exc:  # noqa: BLE001 -- the pool refused the work; say so now
        _finish_job(job_id, "error", error=f"Could not start the analysis worker: {exc}")
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=503, detail="The analysis worker is unavailable; try again."
        ) from exc
    future.add_done_callback(lambda f: _on_job_done(job_id, tmp_path, f))
    # Deliberately still the literal "queued" rather than whatever the job's status has
    # already become. This is an acknowledgement that the work was accepted, and the
    # web app's poller reads it as one; making it race with the worker would turn a
    # constant into a value a client might start branching on.
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
