"""Orchestrates ingestion stage 1 (PRD 2.2): fetch, normalise, and store."""
from __future__ import annotations

from .contracts import Delivery, MatchMeta
from .sources.base import StatsSource
from .storage import ParquetStore


def run_stats_ingestion(source: StatsSource, store: ParquetStore) -> tuple[int, int]:
    """Fetches and normalises one stats source's matches into the warehouse.

    Returns (match_count, delivery_count).
    """
    matches: list[MatchMeta] = []
    deliveries: list[Delivery] = []
    for meta, match_deliveries in source.iter_matches():
        matches.append(meta)
        deliveries.extend(match_deliveries)

    store.write_matches(matches)
    store.write_deliveries(deliveries)
    return len(matches), len(deliveries)
