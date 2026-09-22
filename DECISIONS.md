# Decisions

A running log of deliberate choices made on this project, and why — so a later reader
doesn't have to reverse-engineer intent from a diff. Newest first. Each entry: what was
decided, the alternative(s) considered, and why this one won.

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
