"""CLI entrypoints: `python -m ingestion.cli ingest-stats --competition <slug>`."""
from __future__ import annotations

import argparse

from .pipeline import run_stats_ingestion
from .sources.cricsheet import CricsheetSource
from .storage import ParquetStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="ingestion")
    sub = parser.add_subparsers(dest="command", required=True)

    stats = sub.add_parser("ingest-stats", help="Fetch and normalise a Cricsheet competition into the warehouse.")
    stats.add_argument("--competition", required=True, help="Cricsheet download slug, e.g. icc_mens_t20_world_cup_male")
    stats.add_argument("--cache-dir", default="data/cricsheet")
    stats.add_argument("--warehouse-dir", default="data/warehouse")

    args = parser.parse_args()

    if args.command == "ingest-stats":
        source = CricsheetSource(args.competition, cache_dir=args.cache_dir)
        store = ParquetStore(args.warehouse_dir)
        match_count, delivery_count = run_stats_ingestion(source, store)
        print(f"Ingested {match_count} matches, {delivery_count} deliveries -> {args.warehouse_dir}")


if __name__ == "__main__":
    main()
