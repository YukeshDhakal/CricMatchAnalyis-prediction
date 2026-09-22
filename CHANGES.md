# Changes — for review

What changed in this build pass, grouped by area, for a human review pass. Each item names
the commit(s) it lives in. See DECISIONS.md for the reasoning behind the non-obvious calls,
and MISTAKES.md for what went wrong along the way and how it was caught.

**Test suite status at the end of this pass**: 410 passed, 5 skipped, 4 failed.
The 4 failures are all in `tests/ingestion/test_scene_split_adapter.py` and
`test_manifest_adapter.py` and are environmental — ffmpeg is not on this machine's PATH.
They fail identically on a clean checkout of `11d1854`; this pass added 34 passing tests
(`tests/api/` went from 117 to 151). **Git status**: commits ahead of `origin/main`,
working tree clean, **nothing pushed and nothing deployed** — the running Railway service
is still the previous commit, so the persistence described below is live in this branch and
not yet in production.

---

# Pass 5 (2026-09-23): real accounts, and runs that survive the tab

Third Umpire shipped as a single-tenant showcase: `/app` read shared public tables seeded
from two professional players, and a live run existed only in its own HTTP response — close
the tab and it was gone. This pass makes it a product: self-serve coach/player accounts,
per-account data isolation in Postgres rather than in the browser, and a completed run
written into `video_analyses` owned by whoever ran it.

The web app's half lives in the `third-umpire` repo; this file covers this repo's changes
and the Supabase migration, since that is where the reasoning belongs.

## Supabase: `tu_accounts`, `tu_is_coach()`, and an owner on `video_analyses`
*(this pass)*

Three migrations against project `lshwbwxjlfsjtyzpwuiy`.

- **`public.tu_accounts(user_id pk, account_type, display_name, created_at)`** — one row per
  Third Umpire user. `account_type` is `'coach'` or `'player'` under a CHECK. Policies:
  self-read, coach-read, self-insert. **No update and no delete policy at all**, so a
  session cannot re-type itself after signup; `user_id` being the primary key is what makes
  "exactly one row per user" a key constraint rather than a policy you have to trust.
- **`public.tu_is_coach()`** — `security definer`, `search_path = public`, `EXECUTE`
  revoked from `public`/`anon` and granted to `authenticated`. It only reads, so there is
  no `claim_first_admin`-shaped escalation path through it.
- **`video_analyses.user_id`** — new nullable FK to `auth.users`. The showcase-era
  `public read` policy is dropped (its name was confirmed against `pg_policies` first, not
  guessed), replaced by self-read, coach-read and self-insert.

- **Review this if**: you are wondering why every new policy says `to authenticated`. It is
  load-bearing, and MISTAKES.md has the near-miss. `anon` deliberately cannot execute
  `tu_is_coach()`, so an unscoped policy would make a signed-out read fail with
  `permission denied for function tu_is_coach` instead of returning nothing — leaking the
  shape of the scheme and breaking a public page at the same time.
- The two pre-accounts rows keep `user_id IS NULL`, which no `auth.uid()` can equal, so
  they are visible to coaches only. Nothing was deleted.

## A `tu_accounts` row authorizes a live run
*(this pass)*

`supabase_auth.resolve_caller` reads `tu_accounts` as well as `video_pipeline_access`, with
the caller's own token under RLS, and serves anyone holding either. `SupabaseCaller.role` is
now nullable and there is a new `account_type`.

- **Review this if**: you are checking that this did not widen anybody's access. It did not.
  `wants_frames` still returns true only for `demo`, so a self-serve signup gets no frame
  overlays and no per-track table; a caller holding neither grant is still refused, with a
  403 that does not name which table is missing. The escalation that had to stay closed —
  `tu_accounts` is self-insertable, so a row there must not imply a pipeline role — has its
  own test.
- **Why it was necessary at all**: without it the signup flow led nowhere. MISTAKES.md.
- Both lookups always run rather than short-circuiting on the first grant, so `None` always
  means "holds no such row" and never "wasn't checked". One extra REST call per *cold*
  resolution; the 60-second cache makes those rare.

## A completed run is written into `video_analyses`
*(this pass)*

New `src/api/analysis_store.py`. When a job started by a signed-in user finishes, the
result is `POST`ed to PostgREST **with that user's own bearer token**, so RLS decides the
row's owner rather than this service asserting it — and no service-role key is introduced.

- **Review this if**: you are looking for the new secret. There isn't one. `SUPABASE_URL`
  and `SUPABASE_ANON_KEY` were already set for the bearer-auth work and are the same two
  public values the web app ships to every browser.
- **The cost, stated rather than buried**: the parent process holds the caller's access
  token for the length of their job (`server._job_tokens`), popped exactly once at
  completion and dropped on every terminal path including a dead worker. DECISIONS.md
  weighs that against the alternatives.
- The write happens **before** the job flips to `done`, so the first poll that sees `done`
  carries an accurate `persisted`. A failed write never fails the job: the result comes
  back with `persisted: false` and the reason, which the web app renders as a notice —
  the same rule `pitch_geometry.available` follows.
- The `X-API-Key` path persists nothing (no user to own the row) and says so explicitly,
  rather than leaving a scripted caller to assume its runs are kept.
- Only an explicit allow-list of columns is sent. `timings`, `frame_budget`,
  `pitch_geometry`, `tracks` and `sample_frames` have no column in that table, and
  PostgREST rejects an insert naming a column that does not exist — sending the result
  wholesale would have turned every completed run into an unsaved one.
- **New tests**: `tests/api/test_analysis_store.py` (13) covering the payload projection,
  the header set, every success status PostgREST uses, and above all that nothing here
  raises — this runs on a completion callback, where an escaping exception would leave a
  finished analysis never marked finished. `tests/api/test_server.py` gained 11 more for
  the endpoint behaviour, including that a product account may run, gets no frames, and
  that the held token is dropped on both the success and the failure path.
- **Changed tests**: `test_server.py`'s fixture now stubs `analysis_store.persist_analysis`
  (a unit suite must not write rows into the live project) and its token table carries
  `player`/`coach` callers. `test_supabase_auth.py`'s fake serves the third GET.

## Verified against the live project, not just in tests
*(this pass)*

Two throwaway accounts (one coach, one player) against the real Supabase project, driven
through the real web UI on a local dev server pointed at a locally-run copy of this API.

- A real clip (`real_bowling_clip.mp4`, 48 frames) submitted by the player account through
  the actual `/app/video` form, run through the real pipeline, and **found afterwards as a
  row in `video_analyses` owned by that account** — surviving a reload and a fresh sign-in
  with browser storage cleared, which no run before this pass did.
- RLS proven both ways: the player's REST read returns exactly their own row, the coach's
  returns all three (including the two ownerless demo rows), and a signed-out read returns
  `[]` rather than an error. Reading another account's row by its exact id returns `[]`.
- Escalation probes, all refused: inserting an analysis owned by someone else (403), a
  player promoting itself to coach by UPDATE or by DELETE-and-reinsert (no-ops — the row is
  unchanged), a second account row (primary-key violation), a coach *writing* to a row it
  can read (no-op), an anonymous insert (401), and `anon` calling `tu_is_coach()` by RPC
  (permission denied).
- **Worth knowing**: the signup form was exercised against real Supabase and surfaced its
  real validation errors, but no signup was carried through to a session, because the
  project has email confirmation on and its built-in mailer is rate-limited — completing
  one means mailing a real inbox. The accounts were created the way Supabase's admin API
  creates one instead. Everything after confirmation, including the browser's own
  `tu_accounts` provisioning from signup metadata, is what a confirmed real user hits and
  was exercised for real. MISTAKES.md has the detail.

---

# Pass 4 (2026-09-23): the 4K clip that killed the container

A real 13.7 MB 4K/60fps clip submitted to the live service got a Railway edge
`502 Application failed to respond` mid-job, and the container restarted. The initial
diagnosis blamed GIL starvation — the pipeline ran as a thread inside the uvicorn process.
Measured, that is not what happened: the worst event-loop stall under a pipeline-shaped
CPU-bound thread is 51 ms. The cause was memory, and the metric that was read as ruling
memory out was in fact measuring the bug. See MISTAKES.md.

**The number**: that clip is 598 frames at 3840×2160. `load_frames` decoded every one into
a full-resolution uint8 RGB array and held the list for the whole job — 598 × 23.73 MiB =
**13.86 GiB**, against a reported "peak" of 13.5 GB. A 1083x amplification over the file on
disk, with nothing in a byte-size upload limit able to see it.

## Decoding is bounded in pixels, before the job starts
*(this pass)*

New `src/api/frame_budget.py`. `POST /jobs` probes the clip and refuses it with a 413 if it
is longer than `THIRD_UMPIRE_MAX_FRAMES` (900); `load_frames` gained `max_edge` /
`max_frames` and downscales **during decode** to `THIRD_UMPIRE_MAX_FRAME_EDGE` (1333), so
a full-resolution frame list is never materialised. The incident clip now decodes to
1.67 GiB instead of 13.86 GiB.

- **Review this if**: you are wondering whether this changed anybody's results. 1333 is not
  a round number — it is torchvision `KeypointRCNN`'s own `transform.max_size`, the largest
  edge anything downstream consumes (Ultralytics letterboxes to 640; `overlay` caps at
  960). Footage at or below it is passed through untouched. A test asserts the constant
  equals torchvision's, so the two cannot drift.
- Every result now carries a `frame_budget` block — source resolution, analysed resolution,
  scale, decoded MiB — because a downscaled run returns box coordinates in a different
  space than the caller's source video, and that is not something to leave to inference.
- Over-length clips are **refused, not truncated**. A short read is indistinguishable from
  a short clip at every call site.
- `probe_video` now returns `(width, height, fps, frame_count)`; it reads container
  metadata only and decodes nothing, which is what makes the pre-job check cheap.
- **New tests**: `tests/api/test_frame_budget.py` (13) and
  `tests/video_engine/test_clip_loader_budget.py` (7, 1 opt-in). The second writes real
  video with OpenCV's own encoder and asserts on `nbytes` — the bug lives entirely between
  a file on disk and a list of arrays, so a test starting from arrays cannot see it. Set
  `THIRD_UMPIRE_REAL_4K_CLIP` to run the last one against the actual incident clip.

## The pipeline runs in a child process
*(this pass)*

New `src/api/job_executor.py`. `ThreadPoolExecutor(max_workers=1)` became a
`ProcessPoolExecutor(max_workers=1)` on the `spawn` start method, wrapped so a pool whose
worker died is discarded and rebuilt rather than left permanently broken.
`THIRD_UMPIRE_JOB_EXECUTOR=thread` restores the old behaviour if the process path
misbehaves on the host.

- **Review this if**: you are expecting this to be the fix for the 502. It is not, and
  DECISIONS.md says so — on its own it would have moved the OOM kill from the server to the
  worker and, without the recovery logic, turned one bad clip into a service where every
  later job fails with `BrokenProcessPool`. It is here for blast radius: a segfault or an
  OOM kill in the decode/inference stack now costs one job instead of the service.
- Measured alongside: parent stall over a 5 ms sleep while a worker saturates a core is
  0.50 ms for a process against 10.2 ms for a thread; median HTTP latency under sustained
  load is ~2.4 ms against ~540 ms.
- Job state stays in the parent. The `queued` → `running` transition, which used to be the
  worker's own first act, is now a FIFO in the parent. Temp-file cleanup moved from the
  worker's `finally` to the parent's completion callback — a process killed by the OOM
  killer runs no `finally`.
- **New tests**: `tests/api/test_job_executor.py` (10) with real spawned subprocesses,
  including one that kills a worker with `os._exit` and asserts the next job still runs,
  and one that asserts the worker process is *reused* so the lazy model singleton loads
  once. `tests/api/test_server_responsiveness.py` (4).
- **Changed tests**: `tests/api/test_server.py` now patches `pipeline_runner.run_analysis`
  rather than `server.run_analysis` and runs the executor in thread mode — a monkeypatch
  cannot reach a child process. The worker entrypoint it drives is the shipping one; only
  the executor differs.

## Verified end to end on real 4K footage
*(this pass)*

48 real 3840×2160 frames through the real spawned worker: real YOLO (96 player detections,
mean confidence 0.856; 17 stumps), real ByteTrack (2 players), real Keypoint R-CNN (94 pose
frames), 4 overlay frames and 3 track rows returned across the process boundary. 137 MiB
decoded instead of 1139 MiB; 3.25 GiB peak for the whole run.

- **Worth knowing before re-testing**: pose estimation was **142.9 s of the 154.8 s total**
  — 2.98 s per frame on six CPU threads, and *unaffected by the edge limit*, because
  torchvision resizes to its own `min_size=800` regardless. Extrapolated, the full
  598-frame clip is roughly half an hour here and proportionally less on the hosted box.
  Downscaling saves memory; it does not save pose time. A long-running job is expected, not
  a hang.

---

# Pass 3 (2026-09-22): overlay frames and per-track data out of the API

The web app's live-run panel has never had parity with the Streamlit console, and the two
things it was most obviously missing — frames with the model's boxes drawn on them, and the
per-track table — were missing for a structural reason rather than a UI one: the API had no
way to return either. `run_analysis` returned an aggregate summary and nothing per-frame or
per-track, and the only code that could draw an overlay lived inside `streamlit_app.py`.

## `draw_overlay` moved into the engine
*(`fb71f40`)*

New `src/video_engine/overlay.py`, holding `draw_overlay` verbatim from the Streamlit app
plus `encode_frame`, `sample_overlay_frames` and `tracks_payload`. `streamlit_app.py` now
imports it (and dropped its own `cv2`/`numpy` imports, which nothing else there used), so
there is one implementation rather than two that can drift.

- **Review this if**: you change frame handling anywhere. `io.clip_loader.load_frames`
  returns **RGB**, and `cv2.imencode` is BGR-native. `encode_frame` converts on the way
  out. Omitting that conversion does not raise, does not fail a schema check, and produces
  a perfectly valid JPEG with red and blue swapped in every image.
- Sampled frames are downscaled to a 960px long edge at JPEG quality 80. Measured on
  `real_bowling_clip.mp4`: four 960×540 frames come to ~364KB of base64. At native
  resolution the same four frames are several times that, on a payload that crosses
  Railway → Vercel → browser.
- **New tests**: `tests/video_engine/test_overlay.py` (12). Every one of them decodes the
  base64 back to pixels and asserts on colour and dimensions, because the failure being
  guarded against is an image that is present and wrong, which no "is the field there"
  assertion can see.

## `include_frames` on `POST /jobs`, off by default
*(`fb71f40`)*

`run_analysis` gained `include_frames: bool = False`; when true the result carries `tracks`
and `sample_frames`. The flag is opt-in because everything else in a job result is a fixed
handful of scalars, and these are the only part whose size scales with the footage.

- **Review this if**: you are wondering why the default isn't "always return them". The
  `video_analyses` table has no column for them either, so the default response stays
  exactly the shape that can be inserted directly.
- **New tests**: `tests/api/test_run_analysis_frames.py` (3, CV stack faked), plus 10 new
  cases in `tests/api/test_server.py` covering the flag surviving the HTTP boundary in
  each of the string forms a client might send it as.

## A second video source, and the guard it needs
*(`fb71f40`)*

`POST /jobs` now takes either `file` or `video_url`, matching the Streamlit console's two
sources. New `src/api/video_source.py` is the reason this is a real change rather than a
five-line one.

- **Review this if**: you are tempted to call `ingestion.fetch.download_url` from the API.
  Don't. It is a generic fetcher for trusted callers and speaks `file://` among other
  things; exposing it on a public endpoint is SSRF. The guard enforces an http/https scheme
  allowlist, a video-extension check, a public-address check on **every** resolved address,
  re-validation of **every redirect hop**, and a byte cap enforced on bytes received rather
  than on the `Content-Length` header. Its docstring records the one limit it does not
  close (DNS rebinding between the validating lookup and urllib's own) rather than leaving
  it implied.
- A URL that fails the guard is a 400 on the POST, not a queued job that fails a minute
  later — the caller can fix a URL, and should find out while they are still looking.
- **New tests**: `tests/api/test_video_source.py` (30). The guard is tested against literal
  addresses so no test needs DNS or a network; the download loop is tested against a real
  `ThreadingHTTPServer` on loopback with only the address check neutralised, so the size
  cap and the redirect handler are exercised against real sockets.

## Verified against real footage, not only against fakes

`run_analysis(include_frames=True)` was run on `data/uploads/videos/real_bowling_clip.mp4`
with the real models, and the returned frames were decoded and **looked at**. They show
red player boxes with confidences, yellow pose keypoints on both batsmen, and a green
`ball` box sitting on a light-coloured object in the background — the documented
false-positive behaviour, correctly ending in `shot_classification: unknown` with
`"No ball path, so no pitch geometry."` Colours are correct, which is the BGR/RGB trap
above not firing.

---

# Pass 2 (2026-09-22): ball tracking, trajectory fitting, and line/length

## Ball selection: trajectory fit replaces the pooled-spread motion gate
*(`4b8d3c5`)*

The headline change, and it starts from a correction to this repo's own record. Re-probing
the trained checkpoint over all four clips in `data/uploads/videos/` — and then opening the
frames rather than reading the coordinates — found that the detector's "static false
positives" are frequently **genuine cricket balls lying still on the ground**, detected at
0.44–0.55 across 83 frames of the 4K clip. The real delivered ball scores 0.38–0.65. No
confidence threshold orders those correctly, because a per-frame detector cannot see
motion at all.

It also found a **latent wrong-answer bug in the shipped motion gate**: it pools every ball
detection and measures the spread of the union, so it passes whenever a static object and a
real ball are both present, then reports the release frame of the static one. Measured on
`real_bowling_clip_full.mp4` at a 0.15 confidence threshold: `release_frame=1`, the bag by
the fence. The 0.40 default avoids it only because that bag peaks at 0.34.

- New `src/video_engine/trajectory/` — RANSAC fit selecting a mutually consistent moving
  subset, interpolating frames the detector missed, and locating a bounce by a two-arc
  split or honestly reporting none.
- New `src/video_engine/motion/` — three-frame differencing, scoring how much the pixels
  under a detection actually moved.
- New `src/video_engine/geometry.py` — gives `calibrate_from_stumps` its first caller.
- **Review this if**: you touch ball detection thresholds. The minimum-displacement and
  minimum-step-speed checks are *rejection filters applied before scoring*, not tiebreaks —
  a static cluster outnumbers the real ball 12:1 and wins any inlier-count ranking. That
  ordering is the load-bearing part; see `trajectory/fit.py`'s docstring before changing it.
- **New tests**: `tests/video_engine/test_trajectory.py` (13),
  `test_trajectory_real_footage.py` (4, carrying verbatim recorded detections from the real
  clip), `test_motion_energy.py` (6), `test_geometry.py` (12),
  `test_segmenter_motion_filter.py` (5).

## Motion energy wired in as a pre-filter
*(`0f54497`)*

`EventSegmenter.segment` gained an optional `frames` argument so the segmenter can ask
whether a detection's own pixels were moving. Measured effect on real detections:
`real_bowling_clip_full.mp4` at conf 0.25 goes from 9 candidates to exactly the 7 frames
the real ball occupies; at 0.10, from 59 to 12.

- **Review this if**: you are wondering why this isn't redundant with the trajectory fit.
  It stops a resting ball being absorbed as an *inlier* into a real ball's track, which is
  how contamination appeared at low thresholds — not just a speed optimisation.
- Two refusals are deliberate: the filter disables itself on a panning camera
  (`global_motion_ratio`), and falls back to the unfiltered set when fewer than
  `MIN_INLIERS` survive, because motion energy is meaningless on degenerate input. The
  second was found by `test_pipeline_smoke.py`'s all-zero frames, where a trusting filter
  deleted the entire candidate set.

## Pitch geometry surfaced with its uncertainty attached
*(`459775b`)*

- `DeliveryEvent` gained `pitch_confidence`, `pitch_notes` and `track_confidence`.
- API responses carry a nested `pitch_geometry` block whose `available` flag must be read
  before the numbers; `insufficient_data_reason` is always populated when it is false.
- The Streamlit Video Pipeline Test tab gained a "Line and length" card that renders
  "Insufficient data" and the reason below MEDIUM confidence, rather than a metre figure
  with a caveat under it.
- Also fixes a wrong-answer bug found by running the real pipeline rather than a unit test:
  an undetected bounce was passed through as `bounced=False`, which `classify_length` turns
  into `FULL_TOSS` — reporting a full toss for a ball visibly rolling along the ground. A
  tracking gap is not a cricketing fact; it now reports `UNKNOWN` and says why.
- **Review this if**: you add another consumer of pitch geometry. The rule is that the
  confidence band travels with the number and anything below MEDIUM degrades to
  "insufficient data", matching the existing `ShotType.UNKNOWN` pattern.

## What this pass verified against real footage, and what it did not

Stated separately because the distinction is the point:

- **Verified on real footage**: ball selection. On `real_bowling_clip_full.mp4` the
  pipeline now reports `release_frame=80` (the real ball) where the old gate reported
  frame 1 (the bag), and on `real_bowling_clip3.mp4` / `real_bowling_clip.mp4` it reports
  no track at all rather than a confident wrong one. End-to-end through the real detector,
  tracker and segmenter, not through fixtures.
- **Implemented and unit-tested only**: line, length and the bounce projection. The
  geometry is exercised by synthetic-camera tests and has **never produced a number on real
  footage**, because no available clip shows both stump sets — every clip is filmed from
  behind the bowler's arm, and a homography needs four coplanar points where one stump set
  supplies two. Treat the line/length path as unproven until it runs on suitable footage.
- **Not solved**: identifying *which* moving object is the delivery. On the 4K clip the
  best-scoring track is a white object falling outside the net, fitted cleanly at ~2 px
  RMS — a correct trajectory of the wrong object. Disambiguating it needs the pitch
  corridor, which needs the calibration above, which needs the footage above.

---

# Pass 1: console reskin, warehouse fixes, detector training

## UI: Live Test Console reskin
*(`bfb7d96`)*

- Full dark-theme visual pass across all four tabs (Overview, Player Performance, Video
  Pipeline Test, Rating & Suggestions) — custom CSS, Google Fonts, card/tile components.
- `.streamlit/config.toml` added so Streamlit's own native chrome matches.
- **Review this if**: you want to sanity-check the visual design choices, or you're about to
  touch `app/streamlit_app.py`'s CSS/HTML helper functions (`tu_card`, `_stat_tile_html`,
  etc.) and want to understand the existing pattern first.

## Data: warehouse overwrite bug fixed
*(`bfb7d96`)*

- `ParquetStore.write_matches`/`write_deliveries` now merge/upsert instead of overwriting —
  ingesting a second competition no longer wipes the first.
- `read_matches`/`read_deliveries` gained optional filters (`match_ids`, `competitions`,
  `involving_player`) pushed down via pyarrow, so a scoped read doesn't materialize the
  whole warehouse.
- **Review this if**: you're adding a new ingestion source or another scoped-read caller —
  check `tests/ingestion/test_storage.py`'s "scoped reads" section for the filter semantics
  (the None-vs-empty-list distinction is load-bearing).

## Rating: pipeline performance fix
*(`bfb7d96`)*

- New `src/rating/cohort.py`, `src/rating/pipeline.py`: cohort selection happens before any
  delivery-level read, cutting rating-a-player from 550s+ to ~8.5s against the real 8,026-
  match warehouse.
- **Review this if**: you're changing what counts as a "comparable cohort" — that logic is
  now centralized in `cohort.py` rather than scattered through the rating engine.

## Video engine: pitch calibration groundwork
*(`bfb7d96`)*

- New `src/video_engine/calibration.py`: stump-anchored homography, implemented and tested
  against a synthetic camera, but **has no caller in the pipeline yet** — see its own
  docstring and the README's "Pitch calibration" section for why.
- **Review this if**: you're about to wire calibration into the pipeline — read the
  docstring's honesty section on precision limits (length is more reliable than line) first.

## Video pipeline: URL-based video fetching
*(`6f18713`, carrying a fix from earlier in the session)*

- Video Pipeline Test tab gained a "Fetch from a URL" option alongside file upload.
- `src/ingestion/fetch.py`: `download_url` now sends a real browser User-Agent — several
  real CDNs (GCS buckets, sample-video hosts) 403 the bare `urlretrieve` default.
- **Review this if**: you're adding another network-fetch path — the User-Agent fix applies
  only to this one function; check whether a new caller needs the same treatment.

## Ball/stumps detector: trained and wired, then found unreliable, then partially mitigated
*(`6f18713`, `f3cae57`, `9339429` — read as one arc, not three independent changes)*

1. **Trained** a YOLOv8n on a real, correctly-licensed (CC BY 4.0) 2-class dataset; wired the
   resulting checkpoint (`weights/ball_stumps_n.pt`, gitignored) into `YoloDetector` via
   `app/streamlit_app.py`, with a `THIRD_UMPIRE_BALL_STUMPS_WEIGHTS` env override.
2. **Found**, via real-footage testing requested by the user, that the checkpoint can lock
   onto a static background object at plausible confidence and drive a wrong-but-confident
   shot classification — documented in the README and MISTAKES.md rather than left
   undiscovered.
3. **Mitigated** with a temporal-motion check in `src/video_engine/events/
   heuristic_segmenter.py` that rejects a ball track that doesn't move enough to be real
   flight, pooling detections across track_ids to survive tracking fragmentation. Verified
   against the exact real clip that produced the original wrong answer — same detection
   still fires, now correctly rejected.
- **Review this if**: you're evaluating whether this feature is ready for real use — **it is
  not**. The motion check fixes precision (no more confidently wrong answers) but not recall
  (the real ball is still frequently missed). Read the README's "Ball and stumps detection"
  section in full before deciding whether to build anything on top of this.
- **New tests**: `tests/video_engine/test_heuristic_segmenter.py` (4 tests: no-track,
  rejected-static-track, accepted-moving-track, prefers-moving-over-static).

---

## What's explicitly still open (after pass 2)

- **Footage that shows both stump sets.** The single biggest blocker, and the one only the
  user can clear. Every clip here is filmed from behind the bowler's arm, so only one stump
  set is ever visible and the pitch cannot be calibrated. Line and length are implemented,
  tested and inert until suitable footage exists — a fixed elevated camera with the stumps
  at *both* ends in frame, which is the setup the README's "Pitch calibration" section
  already describes.
- **Footage that shows the ball in flight.** Separate from the above and equally blocking.
  Probing found the delivered ball is recovered by neither the checkpoint nor frame
  differencing on these clips: the deliveries are filmed down the pitch axis, so the ball's
  image displacement is small and heavily blurred, and the bowler's arm dominates the motion
  mask in exactly the corridor the ball travels down. Native-4K differencing did not rescue
  it. Side-on footage would be a far better test of everything built this pass.
- **Telling the delivery from any other moving object.** Currently unsolved and not solvable
  without the calibration above — see the note in the pass-2 section.
- **Ball detection recall.** Unchanged as a model problem: still needs real, non-Roboflow
  training data with motion-blurred balls in flight. What changed is that the pipeline no
  longer converts a recall gap into a confident wrong answer, and the trajectory fit now
  interpolates across frames the detector missed *within* a track it has verified.
- **A retrain.** Not attempted: no labelled data and no GPU training infrastructure in this
  environment. Noted as a deliberate exclusion, not an oversight.
- **`git push`.** 6 commits sit ahead of `origin/main`, unpublished as of this file — left
  for review rather than pushed.
