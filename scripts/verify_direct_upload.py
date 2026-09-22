"""Prove, against the real deployed service, that a clip too big for the old path works.

**The bug this exists to close.** The web app used to upload through a Next.js route on
Vercel, which forwarded the bytes to this API. A Vercel Serverless Function caps the
request body at the *platform gateway*, before the route handler runs, so the upload died
with `413 FUNCTION_PAYLOAD_TOO_LARGE` and the app's own code never executed. Bisected
against the live deployment at the time: a 3.3MB clip passed, a 4.66MB clip did not.
Match footage is nowhere near that small, so the fix was architectural -- the browser now
uploads straight to this container, authenticating with its own Supabase session token
(`api.supabase_auth`) instead of relying on a proxy that held a static key.

"It works now" is a claim about a deployed system, so this script checks it on the
deployed system rather than in a test double. It does the exact sequence a browser does:

    password grant against Supabase  ->  POST /jobs with that bearer token  ->  poll

and it does it with a clip deliberately built to be *larger than the size that used to
fail*. The clip is random noise, which is the point: noise defeats inter-frame
compression, so a few seconds of it is worth many megabytes and the run exercises the
size path honestly rather than shipping a 200KB file and calling the limit tested.

Nothing here is mocked. A pass means a real token from real Supabase Auth carried a real
multi-megabyte upload into the real Railway container.

Run (nothing is read from the environment implicitly -- pass what you mean):

    python scripts/verify_direct_upload.py \
        --api-url https://third-umpire-pipeline-production.up.railway.app \
        --supabase-url https://<project>.supabase.co \
        --supabase-anon-key <anon key> \
        --email <account> --password <password> \
        --size-mb 8

`--no-wait` returns as soon as the job is accepted, which is the part this is really
about; without it the script polls to completion, which takes minutes of real CPU
inference for a clip of this size and is a test of the pipeline rather than of the
upload path.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import requests

try:
    import cv2
except ImportError:  # pragma: no cover -- a clear message beats an ImportError traceback
    print("opencv-python is required to build the test clip: pip install opencv-python")
    raise


def build_noise_clip(dest: Path, target_bytes: int, width: int = 1280, height: int = 720) -> Path:
    """Write an mp4 of random noise until it passes `target_bytes`.

    Frame count is not fixed up front because the encoder's output size per frame is not
    knowable in advance -- it depends on the codec build available on this machine. So
    frames are appended and the file is measured, which is slower and correct, rather
    than estimated and occasionally under target (which would silently turn this into a
    test of a small file).
    """
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a writer for {dest} -- is ffmpeg present?")

    rng = np.random.default_rng(0)
    frames = 0
    try:
        while True:
            for _ in range(25):
                writer.write(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
                frames += 1
            # The file on disk only grows as the encoder flushes, so check periodically
            # rather than per frame, and keep going until it is genuinely over target.
            if dest.exists() and dest.stat().st_size >= target_bytes:
                break
            if frames > 2000:
                raise RuntimeError("gave up building the clip after 2000 frames")
    finally:
        writer.release()

    size = dest.stat().st_size
    if size < target_bytes:
        raise RuntimeError(f"clip finished at {size} bytes, under the {target_bytes} target")
    return dest


def sign_in(supabase_url: str, anon_key: str, email: str, password: str) -> str:
    res = requests.post(
        f"{supabase_url.rstrip('/')}/auth/v1/token",
        params={"grant_type": "password"},
        headers={"apikey": anon_key, "Content-Type": "application/json"},
        json={"email": email, "password": password},
        timeout=20,
    )
    if res.status_code != 200:
        raise SystemExit(f"Supabase sign-in failed ({res.status_code}): {res.text}")
    token = res.json().get("access_token")
    if not token:
        raise SystemExit(f"Supabase returned no access_token: {res.text}")
    return token


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api-url", required=True)
    p.add_argument("--supabase-url", required=True)
    p.add_argument("--supabase-anon-key", required=True)
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    p.add_argument(
        "--size-mb",
        type=float,
        default=8.0,
        help="Clip size floor. Anything above ~4.7 proves the point; 8 is comfortably clear.",
    )
    p.add_argument("--no-wait", action="store_true", help="Stop once the job is accepted.")
    args = p.parse_args()

    api = args.api_url.rstrip("/")

    print(f"1. building a >= {args.size_mb}MB noise clip ...", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="third_umpire_verify_")) / "big_clip.mp4"
    build_noise_clip(tmp, int(args.size_mb * 1024 * 1024))
    size = tmp.stat().st_size
    print(f"   {tmp} -- {size:,} bytes ({size / 1024 / 1024:.2f}MB)")
    if size <= 4_660_000:
        print("   WARNING: that is not larger than the size that used to fail.")

    print("2. signing in to Supabase (password grant, same as the browser) ...", flush=True)
    token = sign_in(args.supabase_url, args.supabase_anon_key, args.email, args.password)
    print(f"   got an access token ({len(token)} chars)")

    print(f"3. POST {api}/jobs with that bearer token and no X-API-Key ...", flush=True)
    started = time.time()
    with tmp.open("rb") as fh:
        res = requests.post(
            f"{api}/jobs",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": (tmp.name, fh, "video/mp4")},
            data={"match_id": "verify-direct-upload", "innings": "1", "over": "0", "ball": "1"},
            timeout=600,
        )
    elapsed = time.time() - started
    print(f"   HTTP {res.status_code} in {elapsed:.1f}s: {res.text[:400]}")

    if res.status_code == 413:
        print("\nFAILED: still rejected for size. Check THIRD_UMPIRE_MAX_UPLOAD_MB on the host.")
        return 1
    if res.status_code != 200:
        print("\nFAILED: the upload was not accepted.")
        return 1

    job_id = res.json()["job_id"]
    print(f"\nPASSED the part that was broken: {size / 1024 / 1024:.2f}MB was accepted (job {job_id}).")

    if args.no_wait:
        return 0

    print("4. polling until the pipeline finishes (real CPU inference -- minutes) ...", flush=True)
    while True:
        poll = requests.get(
            f"{api}/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"}, timeout=30
        )
        if poll.status_code != 200:
            print(f"   poll failed: HTTP {poll.status_code} {poll.text[:300]}")
            return 1
        body = poll.json()
        print(f"   status={body['status']}", flush=True)
        if body["status"] == "done":
            result = body.get("result") or {}
            print(f"\nDONE. frames_processed={result.get('frames_processed')} "
                  f"sample_frames={'yes' if result.get('sample_frames') else 'no'}")
            return 0
        if body["status"] == "error":
            # A pipeline error on random noise is not a failure of *this* test -- the
            # upload path is what was being proved, and it worked.
            print(f"\nThe job errored: {body.get('error')}")
            print("(The upload itself still succeeded, which is what this script tests.)")
            return 0
        time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
