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

**Why an API key is required, not optional.** Unlike the read-only Supabase tables the
web app queries directly, POST /jobs *spends real compute* (and therefore real hosting
cost) on every call. Refusing to start with no key configured, and rejecting every write
without one, is deliberate: an accidentally-public, unauthenticated "run my expensive ML
pipeline" endpoint is a real cost/abuse risk, not just a nice-to-have.

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
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.pipeline_runner import run_analysis
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
) -> dict[str, str]:
    """Starts an analysis job. Supply exactly one video source: an uploaded `file`, or a
    `video_url` this server fetches itself (see `api.video_source` for what it will and
    won't fetch, and why that guard is not optional on a public host).

    `include_frames` opts in to the expensive part of the response -- the per-track table
    and a few real frames with detections and pose keypoints drawn on them. It defaults to
    false so the ordinary response stays small; see `pipeline_runner.run_analysis`.
    """
    _require_api_key(x_api_key)

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
        _jobs[job_id] = Job(id=job_id, status="queued", created_at=time.time(), source_label=source_label)

    _executor.submit(
        _run_job, job_id, tmp_path, source_label, match_id, innings, over, ball, include_frames
    )
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Job:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job.")
    return job
