"""Normalised storage for the ball-by-ball table and match metadata.

Parquet stands in for the PRD's "columnar warehouse" (2.3) at MVP scale --
same columnar-on-disk shape, no server to run. Swap for a real warehouse
(e.g. a managed columnar store) later without changing the write call sites.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .contracts import Delivery, MatchMeta


class ParquetStore:
    def __init__(self, root: Path | str = "data/warehouse"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_matches(self, matches: list[MatchMeta]) -> Path:
        rows = [dict(asdict(m), teams=list(m.teams), dates=list(m.dates)) for m in matches]
        path = self.root / "matches.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        return path

    def write_deliveries(self, deliveries: list[Delivery]) -> Path:
        path = self.root / "deliveries.parquet"
        pd.DataFrame([asdict(d) for d in deliveries]).to_parquet(path, index=False)
        return path

    def read_matches(self) -> pd.DataFrame:
        return pd.read_parquet(self.root / "matches.parquet")

    def read_deliveries(self) -> pd.DataFrame:
        return pd.read_parquet(self.root / "deliveries.parquet")
