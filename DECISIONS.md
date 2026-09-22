# Decisions

A running log of deliberate choices made on this project, and why — so a later reader
doesn't have to reverse-engineer intent from a diff. Newest first. Each entry: what was
decided, the alternative(s) considered, and why this one won.

---

## A `tu_accounts` row authorizes a live run, and `video_pipeline_access` stops being the only gate (2026-09-23)

**Decision**: `supabase_auth.resolve_caller` reads two RLS'd tables instead of one. A
caller is served if they hold *either* a valid `video_pipeline_access` role (the old
hand-issued `admin`/`demo` grant) *or* a `tu_accounts` row (a real coach/player product
account). `SupabaseCaller.role` became nullable and gained `account_type`.

**Why**: without it the product's own signup flow led nowhere. This pass adds self-serve
signup, and a user who created a player account held no `video_pipeline_access` row — so
the one thing the product exists to do, analyse your own footage, was the one thing a new
account could not do. It would have been refused by the same 403 an ungranted stranger
gets. The plan for this pass said `video_pipeline_access` was untouched *and* required a
fresh player account to complete a full live run, which cannot both be true of the old
code; this is the smallest change that makes the second true while keeping the first true
of the table itself, which is not modified at all.

**Alternative considered and rejected**: granting every signup a `video_pipeline_access`
row. That collapses two genuinely different things — "is a customer" and "gets the
showcase account's expensive frame output" — into one, and means every self-serve signup
would have to be hand-provisioned, which is the opposite of self-serve.

**What deliberately did not change**: `wants_frames` still returns true only for `demo`, so
a product account with no hand-issued role gets no frames. Frames are the expensive half of
a response and signing up must not be a way to switch them on for every run on the host.
`tests/api/test_supabase_auth.py::test_a_product_account_does_not_confer_a_pipeline_role`
asserts it, and it is the assertion to leave alone: `tu_accounts` is self-insertable by
design, so if a row there implied any pipeline role, signup would be a privilege-escalation
path.

**Both lookups always run**, rather than short-circuiting once one grants. The resolved
caller is then an unambiguous statement of what the user holds, rather than of where the
lookup happened to stop — which matters because `None` would otherwise mean "no row" or
"not checked" depending on the path. The cost is one extra REST call per *cold* resolution;
the 60-second token cache is what makes those rare.

## A finished run is written back with the caller's own token, not a service-role key (2026-09-23)

**Decision**: new `src/api/analysis_store.py`. When a job started by a signed-in user
completes, the result is `POST`ed to `/rest/v1/video_analyses` with **that user's** bearer
token, so PostgREST evaluates the insert policy (`with check (user_id = auth.uid())`) as
them.

**Why**: two properties fall out of it that a service-role key would have taken away.
Ownership cannot be forged — the row's `user_id` has to equal the token's subject or
Postgres refuses the write, so this service is never in a position to be *wrong* about
whose data something is. And there is no new secret: a service-role key bypasses RLS
entirely, would have to be stored on Railway, and would make this container a much more
interesting thing to compromise. `SUPABASE_URL`/`SUPABASE_ANON_KEY` are the two public
values the web app already ships to every browser.

**The cost, stated plainly**: the parent process holds the caller's access token for the
length of their job, in `server._job_tokens`. That is a real thing to hold and is not done
lightly — the auth cache next door goes out of its way to key on a SHA-256 hash rather than
keep tokens around. Three things made it the better option. The insert has to be made *as
the user*; the moment it has to be made is minutes after the request that carried the token
has already returned; and the alternative, writing on the caller's next poll, makes
persistence depend on a browser staying open, which is exactly the property this pass
exists to remove. It is kept out of the `Job` model (that is serialized into HTTP
responses), popped exactly once at completion, and dropped on every terminal path including
a dead worker and a pool that refused the work.

**A failed write does not fail the job.** The analysis is the expensive part and it
succeeded; discarding minutes of real compute over a retryable database problem would be
the wrong trade. So `persist_analysis` never raises and returns an outcome, the job still
reaches `done`, and the response carries `persisted: false` with the reason — which the web
app renders as a notice on the result. A run that is on screen but not in anyone's history
is one that vanishes on reload, and finding that out *by* reloading is the worse way to
learn it. This is the same rule `pitch_geometry.available` follows: say what is missing
rather than let it look fine.

**The write happens before the job flips to `done`**, not after. The other order leaves a
window where a client is told the run finished and reads a result that does not yet know
whether it was kept. It costs the caller nothing — a job that took minutes is not
meaningfully slower for one bounded REST call.

**Only an explicit allow-list of columns is sent.** A result carries `timings`,
`frame_budget`, `pitch_geometry` and sometimes `tracks` and `sample_frames`, none of which
`video_analyses` has a column for; PostgREST rejects an insert naming a column that does
not exist, so handing over the result dict wholesale would have turned *every* completed
run into an unsaved one.

## The new RLS policies are scoped to `authenticated`, not to `public` (2026-09-23)

**Decision**: every policy added to `tu_accounts` and `video_analyses` is `for select to
authenticated` rather than the unscoped form the plan sketched.

**Why**: `tu_is_coach()` is deliberately not executable by `anon` (`revoke execute … from
public, anon`). A policy that applied to `public` would therefore be *evaluated* for a
signed-out reader, who cannot execute it — and a signed-out `select` on `video_analyses`
would fail with `permission denied for function tu_is_coach` instead of returning nothing.
That would leak the shape of the authorization scheme to anyone who asked, and it would
break "absent looks the same as forbidden", the rule the job-polling endpoint already
follows. Scoped to `authenticated`, an anonymous request matches no policy at all, gets an
empty result, and learns nothing. Verified for real against the live project as both `anon`
and `authenticated`.

## The three showcase screens became honest placeholders, not deletions or stubs (2026-09-23)

**Decision**: Overview, Player performance and Rating & suggestions in the web app now say
"not yet available for your account" and explain why. The tables and the hooks that read
them are kept, unwired.

**Why**: the third option — leaving the professional-player data on screen under a new
signup's name — is the one this project has spent four passes refusing. Those screens are
computed from Cricsheet ball-by-ball data, which has runs, wickets and dismissals in it; an
uploaded clip has none of those. A composite rating for an account with no measurable input
is a number with no measurement under it, which is precisely what the ratings screen's own
"not measured, never zero" pillar treatment was built to prevent. Deleting the tables was
rejected for the opposite reason: they are real output from a real pipeline, the rating
engine still legitimately operates on them, and they are where the deferred aggregation
work plugs back in.

## Account type is self-declared at signup, and confirmation email stays on (2026-09-23)

**Decision**: the signup form asks "I'm a coach" / "I'm a player" and the browser inserts
its own `tu_accounts` row. Supabase's standard confirmation-email flow is left enabled.

**Why (confirmation)**: turning it off is an intentional abuse tradeoff on a service where
every account can spend real metered CPU, not a default to take for frictionlessness.

**The consequence that shaped the code**: with confirmation on, `signUp` returns a user and
**no session**, so the browser has no `auth.uid()` to insert the `tu_accounts` row under
and the insert policy correctly refuses. The chosen type and name therefore travel in
`options.data` (Supabase user metadata) and are turned into a row on the first request that
actually holds a session — after the confirmation link, on another device, whenever. That
metadata is user-writable, which sounds worse than it is: the only thing it can do is
choose the type of a row the same user was already free to choose the type of at signup,
and once the row exists nothing writes to it again, because `tu_accounts` has no update or
delete policy at all.

**Flagged rather than solved**: "coach" being self-declared means anyone who signs up as a
coach can read every player's run history. That is the product model as specified — a
coach↔player connection flow was explicitly ruled out — and it is fine while the accounts
are the owner's own. It is the thing to revisit before the signup link is public, and it is
written into both READMEs rather than left in a commit message.

## A coach can read `tu_accounts`, not just their own row (2026-09-23)

**Decision**: a third policy, `accounts readable by coaches`, using `tu_is_coach()`.

**Why**: the plan's own coach player-switcher is described as being populated from
"`user_id`/`display_name` pairs the coach can see", and self-read alone supplies no name
for anybody else — `video_analyses` carries a `user_id` and nothing human-readable. Without
this the switcher would have listed truncated UUIDs. It exposes `display_name` and
`account_type`, which is everything `tu_accounts` holds, to a role that can already read
every one of those users' analyses.

**Checked rather than assumed**: a select policy on `tu_accounts` that calls a function
which itself reads `tu_accounts` is a textbook infinite-recursion setup. It does not recur
here because `tu_is_coach()` is `security definer` and owned by `postgres`, which owns the
table and is not under `FORCE ROW LEVEL SECURITY`, so the inner read bypasses RLS. That was
verified by querying as a simulated `authenticated` session, not reasoned about.

---

## The upload limit is expressed in decoded pixels, not in bytes on the wire (2026-09-23)

**Decision**: `POST /jobs` probes every clip before creating a job and refuses it (413) if
its frame count exceeds `THIRD_UMPIRE_MAX_FRAMES`; `load_frames` downscales during decode
so no frame's longest edge exceeds `THIRD_UMPIRE_MAX_FRAME_EDGE` (1333), and enforces the
frame cap itself while reading. `THIRD_UMPIRE_MAX_UPLOAD_MB` stays, but it is no longer the
limit that matters. See `src/api/frame_budget.py`.

**Why**: measured, on the clip that took production down. 13.1 MiB on disk, 598 frames at
3840×2160, **13.86 GiB** decoded and held for the length of the job — a 1083x amplification
that is a property of the codec and the footage, not of the request. A byte-size limit
cannot see any of it, which is how a 50 MB ceiling came to accept clips that no machine
this will ever run on could decode.

**Why 1333 specifically**: it is the largest edge anything downstream actually consumes.
Ultralytics letterboxes to `imgsz=640` before YOLO's forward pass; torchvision's
`KeypointRCNN_ResNet50_FPN` transform resizes to `min_size=800, max_size=1333`;
`overlay.DEFAULT_MAX_EDGE` caps returned frames at 960. The pipeline was decoding, copying
and holding 13.8 GiB of pixels that no model ever read. Footage at or below 1333 is passed
through untouched, so every clip that worked before is analysed bit-identically — a fix
that silently changed results for working footage would be trading one bug for a quieter
one. `tests/api/test_frame_budget.py` pins the constant to torchvision's own `max_size` so
the two cannot drift apart unnoticed.

**What this does change, and it is written into every response**: downscaling shrinks box
coordinates by the scale factor. Nearly every spatial threshold in `video_engine` is
already a multiple of the ball box (`TOLERANCE_BOX_MULTIPLE`, `MIN_DISPLACEMENT_BOX_MULTIPLE`,
`MIN_STEP_SPEED_BOX_MULTIPLE`, `CLUSTER_RADIUS_MULTIPLE`, `MIN_END_SIZE_RATIO`) and
`motion.energy` works in fractions of changed pixels, so those are scale-invariant. Two are
absolute pixels — `trajectory.fit.MIN_TOLERANCE_PX` (4.0) and
`calibration.MAX_REPROJECTION_ERROR_PX` (2.0). Both bound *error*, and error shrinks with
the coordinates, so both become relatively more permissive on downscaled footage. That is
the safer direction, but it is a real difference, so every result now carries a
`frame_budget` block naming the source and analysed resolutions and the scale between them.

**Alternatives rejected**:

- *Subsample frames instead of resizing.* Dropping every second frame would halve memory
  too, and would break the things that depend on consecutive frames: `motion.energy`'s
  three-frame window and `trajectory.fit`'s `MAX_FRAME_GAP`. Resolution is the axis nothing
  downstream is using; frame cadence is one everything is.
- *Truncate over-length clips to the first N frames.* A short read is indistinguishable
  from a short clip at every call site. "The ball was never released" and "the release was
  in the frames we threw away" would be the same answer. Over-length footage is refused
  with a message saying so.
- *Raise the container's memory limit.* Buys one clip's worth of headroom and nothing else.
  A 20-second 4K clip is 27.7 GiB; the amplification is unbounded in duration, so the only
  limit that holds is one on what gets decoded.

**Why the frame cap is 900 and not larger**: memory would allow far more — 900 frames at
1333px is about 2.6 GiB. The binding constraint is Keypoint R-CNN, measured at 2.98 s per
frame on six CPU threads and unaffected by the edge limit (torchvision resizes to its own
`min_size` regardless). 900 frames is 15 seconds at 60fps, accepts the 598-frame incident
clip with room, and keeps the worst case from being an hour-long job nobody is still
polling for.

---

## The pipeline runs in a child process — for blast radius, not for latency (2026-09-23)

**Decision**: `api.server`'s single worker moved from `ThreadPoolExecutor(max_workers=1)` to
a `ProcessPoolExecutor(max_workers=1)` using the `spawn` start method, wrapped in
`api.job_executor.JobExecutor` so a pool whose worker died is discarded and rebuilt.
`THIRD_UMPIRE_JOB_EXECUTOR=thread` restores the old behaviour.

**Why not the reason it was first proposed**: the original theory was that the thread was
starving uvicorn's event loop through the GIL until Railway's edge timed out. Measured, the
worst event-loop stall under a pipeline-shaped CPU-bound thread is 51 ms. That is not how a
container gets killed, and the real cause was memory (see the entry above and MISTAKES.md).
This change is second in line and would not have fixed the incident on its own.

**Why do it anyway**: everything the worker runs is native code operating on
attacker-supplied video — OpenCV decoders, Ultralytics, torch. When something in that stack
dies hard, a thread takes the HTTP server down with it because they share an address space.
In a child, the same death is one failed job with a real message, and the next job gets a
fresh worker. There is also a measured latency benefit that is larger than expected: with a
worker saturating a core, the parent's stall over a 5 ms sleep is 0.50 ms against a process
and 10.2 ms against a thread, and median HTTP latency under sustained load is ~2.4 ms
against ~540 ms.

**Why `spawn` on Linux too**, where `fork` is the default: the pool's worker is created
lazily on first submission, long after uvicorn has started its threads, and forking a
multi-threaded process hands the child locks held by threads that do not exist in it —
torch's OpenMP pool is a well-known way to deadlock exactly like that. `forkserver` avoids
it but is POSIX-only, and shipping a start method the Windows test suite never exercises
would mean the thing tested is not the thing deployed. The cost is that the child re-imports
torch and reloads the models on its first job; that is per *worker*, not per job, and
`tests/api/test_job_executor.py` asserts the reuse rather than assuming it.

**The failure mode that had to be handled first**: `ProcessPoolExecutor` is permanently
unusable after a worker exits uncleanly — every pending future and every later `submit`
raises `BrokenProcessPool`. Adopting it without handling that would have turned one killed
worker into a service where every subsequent job fails until someone restarts the container:
strictly worse than the thread pool, which at least died visibly and came back. Two tests
cover it, one killing the worker with `os._exit` and asserting the next job runs.

**Consequences worked through**: job state (`_jobs`, `_jobs_lock`) stays in the parent,
which is where the HTTP handlers read it; the worker never sees it. The `queued` → `running`
transition used to be the worker's own first act and is now a FIFO in the parent, which with
one worker is exact. Deleting the uploaded temp file moved from the worker's `finally` to
the parent's completion callback, because a process killed by the OOM killer runs no
`finally` and the file would leak on precisely the failures most likely to repeat. Failures
are flattened to a string *in the child* rather than raised across the boundary, because a
good deal of what this stack raises does not survive pickling.

**Alternatives rejected**:

- *Yield to the event loop periodically inside the pipeline.* Invasive, touches every
  module, and aimed at a problem the measurements say does not exist.
- *A separate worker container behind a queue.* Correct at a larger scale and a much bigger
  infrastructure change — a queue, a second Railway service, shared storage for uploads —
  for a job volume that is one clip at a time.

---

## Overlay frames are opt-in per request, not part of every job result (2026-09-22)

**Decision**: `run_analysis` and `POST /jobs` take `include_frames`, defaulting to false.
When false the result is exactly what it was before; when true it also carries `tracks` and
a few base64 JPEG frames with detections and pose keypoints drawn on them.

**Why**: every other field in a job result is a scalar, and the whole response is around a
kilobyte. Sampled frames are measured at ~364KB for four 960×540 frames from
`real_bowling_clip.mp4`, and that number is a property of the footage rather than of the
schema. The consumer that wants them (the web app's demo role) is one of several, the
`video_analyses` table cannot store them, and a response whose size varies by three orders
of magnitude depending on the clip is a worse default than one that doesn't.

**Alternatives rejected**:

- *Always return them.* Makes the cheap path pay for the expensive one, and breaks the
  property that a job result is insertable into `video_analyses` as-is.
- *A separate `GET /jobs/{id}/frames` endpoint.* Cleaner in isolation, but the frames only
  exist while the analysis is running — the decoded frames are not retained after
  `run_analysis` returns, so a second endpoint would mean either holding every job's frames
  in memory or re-running the pipeline. Deciding at submit time is what avoids both.

**What the flag deliberately does not do**: it is not an authorisation mechanism. The API
still requires its key for any `POST /jobs`; `include_frames` only shapes the response. The
decision about *who* may ask for frames is made in the web app's proxy, which is the layer
that knows who the user is.

---

## A guarded fetcher for `video_url`, rather than reusing `ingestion.fetch.download_url` (2026-09-22)

**Decision**: `POST /jobs` accepts a `video_url`, but fetches it through a new
`src/api/video_source.py` rather than through the existing `download_url` that the CLI and
the Streamlit console both use.

**Why**: the two callers have different threat models and the same function cannot serve
both honestly. Streamlit runs on the developer's machine, where "fetch the URL I typed" is
the entire security question. The API is reachable from the internet, so the same feature
means *an untrusted caller choosing an address the server will connect to* — server-side
request forgery. `urllib` will happily open `file:///etc/passwd`, and a container host's
metadata endpoint sits on a link-local address that looks like an ordinary URL.
`download_url` cannot defend against that and should not try: it is used for stats files by
trusted callers, and adding address filtering there would impose a network policy on code
that has no reason to carry one.

**Alternatives rejected**:

- *Validate in the Next.js proxy instead.* Moves the same problem to a different server. The
  fetch happens here, so the guard belongs here — and the API is callable without going
  through the web app at all.
- *Accept only an allowlist of hosts.* Too narrow to be useful for the actual use case
  (arbitrary broadcast CDN and S3 links) and no stronger than an address check, since a
  host on the allowlist can still redirect.
- *Have the browser fetch and upload the bytes.* Defeats the point of a URL source and runs
  into Vercel's request-body limits for anything but a small clip.

**The parts that are easy to leave out and are load-bearing**: *every* resolved address is
checked, not just the first; *every* redirect hop is re-validated, because urllib follows
redirects by default and a check on only the submitted URL is no check at all; and the byte
cap counts bytes received rather than trusting `Content-Length`, which a hostile server
writes. The residual DNS-rebinding window is recorded in the module docstring rather than
quietly accepted.

---

## Select the ball by trajectory consistency, not by confidence or pooled spread (2026-09-22)

**Decision**: replace the pooled-displacement gate with a RANSAC fit
(`video_engine/trajectory/fit.py`) that searches for a subset of the pooled BALL
detections lying on one physically plausible path, and treat that subset — not the union —
as the ball track.

**Why**: re-probing all four real clips established that the discriminating signal is not
confidence and not aggregate motion, but *joint consistency*. Three measurements drove it:

- A genuine cricket ball resting on the outfield scores 0.55 across 83 frames while the
  real delivered ball scores 0.38–0.65. **No confidence threshold separates them**, and
  raising the threshold to suppress the resting ball also suppresses the real one.
- The pooled-spread gate passes whenever two objects are present, because the spread it
  measures is the distance between them. It then reports the wrong release frame. See
  MISTAKES.md.
- A static cluster outnumbers the real ball roughly 12:1 on the 4K clip, so any scheme that
  ranks candidate tracks by support alone selects the wrong one.

**Alternatives rejected**:

- *Lower the detector threshold to fix recall.* Measured: it does raise recall, but the
  additional detections at 0.05–0.15 are overwhelmingly more static clutter, so precision
  falls faster than recall rises. It is only safe once something downstream can reject
  clutter structurally — which is what this change provides, and why the two go together.
- *Retrain.* No new labelled data and no GPU training infrastructure in this environment,
  and the dominant failure is not a labelling error the model could learn away: the
  resting balls are correctly labelled balls.
- *Kalman filter / ByteTrack alone.* A tracker smooths and associates but does not decide
  which of several genuinely-tracked objects is the delivery, which is the actual question.

**The ordering that makes it work, and is easy to get wrong**: minimum displacement and
minimum median step speed are **rejection filters applied before scoring**, not tiebreaks
applied after. A stationary cluster fits a zero-velocity path perfectly and has more
members than the real ball, so under any "most inliers wins" ranking it wins. Hypotheses
that do not move are not weak candidates; they are not candidates.

**Trade-off accepted**: a ball track now needs at least five detections before anything is
reported, where two used to suffice. Three points lie exactly on some parabola and
therefore verify nothing, so the old bar was not evidence at all — but this does mean more
clips report nothing, and that is the intended direction.

**What it explicitly does not solve**: identifying *which* moving object is the delivery.
On the 4K clip the best-scoring track is a white object falling outside the net, fitted
cleanly at ~2 px RMS. That is a correct trajectory of the wrong object, and no amount of
trajectory reasoning fixes it — it needs the pitch corridor, i.e. a calibration, i.e.
footage showing both stump sets. The step-speed threshold was deliberately **not** tuned
upward to exclude that object, because a threshold that excludes the wrong answer by a
hair is the trap the previous gate fell into.

---

## Give `calibrate_from_stumps` a caller by making its refused assumption explicit (2026-09-22)

**Decision**: add `video_engine/geometry.py` as the caller `calibration.py` has lacked
since it was written, and resolve the striker's/bowler's end question by applying the
near/far size heuristic **only when it is decisive**, returning the assumption as text that
travels with the result into `DeliveryEvent.pitch_notes` and the API response.

**Why this doesn't contradict the earlier decision to refuse**: `calibrate_from_stumps`
declines to infer which end is which, on the grounds that "the larger box is nearer" is
*right often enough to be dangerous*. That reasoning is about a *silent* inference. The
danger is not the heuristic, it is a length measured from the wrong end with nothing in the
output saying so. Three things remove that danger without leaving the module permanently
uncallable: the assignment is refused unless the two sets differ by at least 1.5x in
apparent height (side-on, where the heuristic has no signal, still refuses), the assumption
is carried in the output a user reads, and the confidence band is capped below `HIGH`.

**Only the bounce point is projected, never the flight.** The homography maps the pitch
*plane*. A ball in flight is above it, so unprojecting a mid-flight detection produces a
confident number with no physical meaning. At ground contact the ball is on the plane by
definition. That single point is the only place the mapping is exact, which is why the
trajectory module's job ends at locating a bounce and the geometry module's begins there.

**Consequence accepted, and it is the honest headline**: no clip currently in the repo
shows both stump sets — the camera sits behind the bowler's arm in all of them — so this
path returns `GeometryConfidence.NONE` on every piece of real footage available. Line and
length are wired, tested and inert, and the thing that unblocks them is footage, not code.

---

## Reject false-positive ball detections via a motion check, not a retrain (2026-09-17)

**Decision**: guard `HeuristicEventSegmenter` against a bad ball detection by requiring a
candidate ball track's pooled detections to move at least `3x` their own bounding-box size
across the clip, rather than retraining the checkpoint or rewriting detection entirely.

**Why**: real-footage testing (three clips, see MISTAKES.md) found the trained ball+stumps
checkpoint can lock onto a static background object with plausible confidence, producing a
wrong-but-confident shot classification. Retraining needs better data that wasn't available
in-session; a full tracking rewrite (background subtraction, Kalman filtering) is real work
with no guaranteed payoff. A motion check is cheap, ships immediately, and directly closes
the *specific, demonstrated* failure mode: a real ball crosses most of the frame during a
delivery, a static false positive doesn't move beyond ordinary inference jitter.

**Trade-off accepted**: this fixes precision (no more confidently wrong answers) but not
recall (the checkpoint still often doesn't find the real ball at all). `unknown` remains the
common, honest outcome until a better detector exists.

**Implementation detail worth remembering**: the motion check pools ball detections across
*every* track_id before measuring displacement, rather than checking motion within a single
track. IOU-based tracking (and ByteTrack under fast motion) fragments a genuinely fast ball
into several short-lived tracks with near-zero frame-to-frame overlap — checking motion
per-track rejected real balls for the same reason it correctly rejected static ones. Caught
by `tests/test_pipeline_smoke.py` failing against the first, per-track version of this fix.

---

## Train a 2-class (ball, stumps) YOLOv8n from scratch on a CC BY 4.0 dataset (2026-09-16/17)

**Decision**: use Roboflow's public "Cricket Dataset" (workspace `yukeee`, project
`cricket-dataset-z2wkt-wcfjg`, 7,452 images, CC BY 4.0) to fine-tune YOLOv8n, rather than
using either of the two public alternatives surveyed.

**Alternatives rejected**:
- `sanjusabu/Cricket-Ball-and-Stumps-Detection` — no dataset, no weights, no licence. A
  README with nothing behind it.
- `kushagra3204/Cricket-Ball-Trajectory-Prediction` — a real, genuinely fine-tuned ball-only
  detector, but no LICENSE file (default copyright, not redistributable) and ball-only (no
  stumps class, which pitch calibration needs).

**Why train from scratch instead of finding a third option**: the two realistic routes were
(a) get the `kushagra3204` author's explicit permission and separately source/annotate a
stumps dataset, or (b) train a 2-class model from a licensed dataset. (b) was faster once a
correctly-licensed, correctly-labeled dataset was actually found, and sidesteps an
open-ended permission-seeking dependency on someone else's goodwill.

**Where training happened**: local CPU training was measured at ~16 hours for the full run
and caused real system memory pressure — moved to Google Colab's free T4 GPU tier (0.911
hours wall-clock for 30 epochs). See MISTAKES.md for what went wrong along the way.

---

## Merge/append the Parquet warehouse instead of migrating to a database engine (session predates this file)

**Decision**: fix `ParquetStore.write_matches`/`write_deliveries` to read-existing,
concat, `drop_duplicates(keep="last")`, write — rather than swap Parquet for SQLite/DuckDB/
Postgres.

**Why**: the bug (ingesting a second competition silently wiped the first) was a plain
`to_parquet` overwrite, not a fundamental limitation of the file format. A read-modify-write
upsert fixes it directly; a storage-engine migration would have been a much larger change
for a problem that didn't need one. `read_matches`/`read_deliveries` grew optional filters
pushed down via pyarrow (`match_ids`, `competitions`, `involving_player`) so a scoped read
still costs proportional to what's asked for, not the whole warehouse — the main practical
downside a plain-Parquet approach would otherwise have at multi-competition scale.

---

## Cohort-scoped rating pipeline instead of fusing the whole warehouse (session predates this file)

**Decision**: `rating.pipeline` selects a comparable cohort and the player's last-N window
first, reads only those rows (via the scoped-read filters above), pools the baseline
vectorised, and fuses only the player's window — rather than fusing the entire warehouse
and filtering afterward.

**Why**: the old order was O(matches × deliveries), measured at 550s+ against the real
8,026-match warehouse (effectively unusable). Selecting the cohort *before* touching
delivery-level data turns that into a scoped read plus one vectorised pass, measured at
8.5s on the same data — the ordering, not a smarter algorithm on the same data shape, is
what fixed it.
