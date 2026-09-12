"""CLI entrypoints:

  python -m ingestion.cli ingest-stats --competition <slug>
  python -m ingestion.cli fetch-url --url <link> --dest videos
  python -m ingestion.cli ingest-video --match-id <id> --video <path> --innings 1 --over 0 --ball 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fetch import download_url
from .pipeline import run_stats_ingestion
from .sources.cricsheet import CricsheetSource
from .storage import ParquetStore
from .video.manifest_adapter import ManifestClipAdapter


def main() -> None:
    parser = argparse.ArgumentParser(prog="ingestion")
    sub = parser.add_subparsers(dest="command", required=True)

    stats = sub.add_parser("ingest-stats", help="Fetch and normalise a Cricsheet competition into the warehouse.")
    stats.add_argument("--competition", required=True, help="Cricsheet download slug, e.g. icc_mens_t20_world_cup_male")
    stats.add_argument("--cache-dir", default="data/cricsheet")
    stats.add_argument("--warehouse-dir", default="data/warehouse")

    fetch = sub.add_parser("fetch-url", help="Download a direct file link (video or stats file) into data/uploads/<dest>.")
    fetch.add_argument("--url", required=True)
    fetch.add_argument("--dest", choices=["videos", "stats"], required=True)

    video = sub.add_parser("ingest-video", help="Wrap one local video file into a DeliveryClip.")
    video.add_argument("--match-id", required=True)
    video.add_argument("--video", required=True, help="Path to a single delivery's video file.")
    video.add_argument("--innings", type=int, required=True)
    video.add_argument("--over", type=int, required=True)
    video.add_argument("--ball", type=int, required=True)

    args = parser.parse_args()

    if args.command == "ingest-stats":
        source = CricsheetSource(args.competition, cache_dir=args.cache_dir)
        store = ParquetStore(args.warehouse_dir)
        match_count, delivery_count = run_stats_ingestion(source, store)
        print(f"Ingested {match_count} matches, {delivery_count} deliveries -> {args.warehouse_dir}")

    elif args.command == "fetch-url":
        dest_dir = Path("data/uploads") / args.dest
        path = download_url(args.url, dest_dir)
        print(f"Downloaded -> {path}")

    elif args.command == "ingest-video":
        video_path = Path(args.video)
        manifest_path = video_path.parent / f".{video_path.stem}.manifest.json"
        manifest_path.write_text(
            json.dumps({video_path.name: {"innings": args.innings, "over": args.over, "ball": args.ball}}),
            encoding="utf-8",
        )
        clip = ManifestClipAdapter().ingest(args.match_id, video_path.parent, manifest_path)[0]
        print(f"DeliveryClip: {clip.delivery} video_path={clip.video_path} "
              f"{clip.width}x{clip.height} @ {clip.fps:.1f}fps source={clip.source.value}")


if __name__ == "__main__":
    main()
