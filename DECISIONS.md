# Decisions

A running log of deliberate choices made on this project, and why — so a later reader
doesn't have to reverse-engineer intent from a diff. Newest first. Each entry: what was
decided, the alternative(s) considered, and why this one won.

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
