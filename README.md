# Third Umpire

An AI cricket analyst: it reads match video and official T20 stat tables, and returns a
per-player rating with clip-referenced coaching suggestions. Product and architecture
context lives in the team's PRD; this repo is the implementation.

## Modules

- **`video_engine/`** (this repo, in progress) — turns a per-delivery video clip into
  tracked objects, player poses, and a segmented event (shot type, release/contact frames).
- **Ingestion** (owned by another contributor) — produces the `DeliveryClip` this module
  consumes: see "Contract with ingestion" below.
- **Fusion / rating / suggestion engine** — not started yet; consumes `DeliveryAnalysis`
  from `video_engine` alongside the official ball-by-ball table.

## Contract with ingestion

The video engine assumes ingestion has already split match footage into one short clip
per delivery, and hands off a `DeliveryClip` (`src/video_engine/contracts.py`):

| Field | Type | Notes |
|---|---|---|
| `delivery.match_id` | `str` | Joins back to the official ball-by-ball table |
| `delivery.innings` / `.over` / `.ball` | `int` | Ball number as bowled (wides/no-balls included) |
| `video_path` | `str` | Path to the delivery's video file |
| `fps`, `width`, `height` | `float` / `int` | Source video properties |
| `source` | `"broadcast"` \| `"user_upload"` | Broadcast footage vs. a phone/user upload |

If ingestion's actual output looks different (e.g. raw frame arrays instead of a video
file, or a different metadata shape), update `DeliveryClip` and `load_frames` in
`src/video_engine/io/clip_loader.py` rather than adapting around the mismatch elsewhere —
this is the one place that boundary is meant to live.

## Video engine pipeline

Four stages, in dependency order — each one built to unblock the next:

1. **Detection** (`detection/`) — player, ball, and stumps bounding boxes per frame.
   `YoloDetector` uses a pretrained YOLO model for players; ball/stumps detection needs a
   checkpoint fine-tuned on cricket footage (`ball_stumps_weights`), which doesn't exist
   yet — pass `None` and it will emit player detections only, with a logged warning. Runs
   each model once over the whole clip (`batch_size` frames per forward pass, default 16)
   rather than once per frame.
2. **Tracking** (`tracking/`) — links per-frame detections into persistent tracks.
   `ByteTrackTracker` (recommended default) wraps a real Kalman-filter multi-object tracker
   — it predicts through brief occlusions and handles the ball's erratic, fast motion far
   better than IOU matching alone. `IouTracker` is kept as a zero-dependency fallback for
   quick local tests.
3. **Pose estimation** (`pose/`) — `KeypointRcnnPoseEstimator` runs a pretrained
   Keypoint R-CNN and matches each detected pose onto the nearest player track by box
   overlap. Batches `batch_size` frames per forward pass (default 8) instead of one at a time.
4. **Event segmentation** (`events/`) — `HeuristicEventSegmenter` finds the release frame
   (first ball detection) and contact frame (closest ball/player approach), then classifies
   the shot from the batter's wrist swing. This is a rule-based v0; replace it with a
   trained classifier once labelled clips exist.

`pipeline.VideoEngine` wires all four into `analyze(clip) -> DeliveryAnalysis`.

## Security notes

- **`torch>=2.6` is a pinned floor, not just a version bump.** That release switched
  `torch.load`'s default to `weights_only=True`, which stops a malicious or corrupted
  `.pt` checkpoint from executing arbitrary code via pickle on load. Don't downgrade
  below it, and don't override `weights_only` when loading a checkpoint from outside
  the team (e.g. a `ball_stumps_weights` file someone hands you).
- **Only load model weights from a source you trust** — a teammate's checkpoint, a
  pinned Ultralytics release asset, or your own training run. Treat an unfamiliar `.pt`
  file the same as you'd treat unfamiliar executable code.
- **All dependencies are version-range pinned** (`pyproject.toml` / `requirements.txt`)
  rather than left unbounded, so a future release can't silently change behavior or pull
  in a compromised transitive dependency without the range being revisited.
- **`video_path` is trusted input for now.** `load_frames` opens whatever path
  `DeliveryClip` gives it with no sandboxing. That's fine while ingestion and the video
  engine run in the same trusted pipeline; if a future API ever accepts a path from an
  external caller, validate/sandbox it there before it reaches this module.

## Known gaps (tracked, not blocking)

- No fine-tuned ball/stumps detector yet — needs a labelled cricket dataset.
- The "batter" in a clip is inferred as whichever player track is nearest the ball at
  contact; there's no explicit role labelling (batter vs. bowler vs. fielder) yet.
- Shot classification is a hand-set heuristic on wrist displacement, not learned.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests run against fakes/stubs for the model-backed stages (`Detector`, `PoseEstimator`),
so they don't need model downloads or a GPU. `IouTracker`, `ByteTrackTracker`, and
`HeuristicEventSegmenter` are pure CPU logic (no model weights) and are tested directly.
