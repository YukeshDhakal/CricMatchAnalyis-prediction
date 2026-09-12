"""Cricsheet (cricsheet.org) stats source.

Cricsheet publishes free, structured ball-by-ball data for international and
T20-league cricket under an open license (see cricsheet.org/about/#data_use).
It stands in for the PRD's licensed "ICC / board ball-by-ball tables" source
(2.1) for development and for any competition it actually covers -- swap in a
licensed-feed StatsSource once a data-partner agreement is in place; nothing
downstream needs to change since both implement the same StatsSource interface.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Iterator, Optional
from urllib.request import Request, urlopen, urlretrieve

from ..contracts import Delivery, MatchMeta
from ..rules import phase_for_over
from .base import StatsSource

CRICSHEET_BASE_URL = "https://cricsheet.org/downloads"


class CricsheetSource(StatsSource):
    def __init__(self, competition: str, cache_dir: Path | str = "data/cricsheet"):
        """`competition` is a Cricsheet download slug, e.g. 'icc_mens_t20_world_cup_male', 't20s_male', 'ipl_male'."""
        self.competition = competition
        self.cache_dir = Path(cache_dir)
        self.extract_dir = self.cache_dir / competition
        self._meta_path = self.cache_dir / f"{competition}.meta.json"

    def fetch(self, force: bool = False) -> Path:
        """Re-downloads whenever cricsheet.org's copy has actually changed.

        Every call checks the live `Last-Modified` header against what was recorded
        on the last successful download -- this is a HEAD request, not a full
        download, so calling it often is cheap. Cricsheet's archives are updated
        periodically (new matches added), not streamed, so "live" for this source
        means always reflecting its current state rather than a local snapshot
        frozen at whatever point it happened to be first fetched.
        """
        url = f"{CRICSHEET_BASE_URL}/{self.competition}_json.zip"
        remote_last_modified = self._remote_last_modified(url)
        up_to_date = (
            not force
            and self.extract_dir.exists()
            and remote_last_modified is not None
            and remote_last_modified == self._read_cached_last_modified()
        )
        if up_to_date:
            return self.extract_dir

        if self.extract_dir.exists():
            shutil.rmtree(self.extract_dir)
        self.extract_dir.mkdir(parents=True, exist_ok=True)

        zip_path = self.cache_dir / f"{self.competition}_json.zip"
        urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(self.extract_dir)
        zip_path.unlink()

        self._write_cached_last_modified(remote_last_modified)
        return self.extract_dir

    def iter_matches(self) -> Iterator[tuple[MatchMeta, list[Delivery]]]:
        self.fetch()  # always re-checks the live source first, even if already cached
        for path in sorted(self.extract_dir.glob("*.json")):
            with path.open(encoding="utf-8") as f:
                raw = json.load(f)
            yield parse_match(path.stem, raw)

    def _remote_last_modified(self, url: str) -> Optional[str]:
        with urlopen(Request(url, method="HEAD")) as response:
            return response.headers.get("Last-Modified")

    def _read_cached_last_modified(self) -> Optional[str]:
        if not self._meta_path.exists():
            return None
        return json.loads(self._meta_path.read_text(encoding="utf-8")).get("last_modified")

    def _write_cached_last_modified(self, value: Optional[str]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(json.dumps({"last_modified": value}), encoding="utf-8")


def parse_match(match_id: str, raw: dict) -> tuple[MatchMeta, list[Delivery]]:
    info = raw["info"]
    teams = tuple(info["teams"])
    event = info.get("event") or {}
    outcome = info.get("outcome") or {}
    toss = info.get("toss") or {}

    meta = MatchMeta(
        match_id=match_id,
        competition=event.get("name", info.get("match_type", "unknown")),
        match_type=info.get("match_type", ""),
        gender=info.get("gender", ""),
        teams=teams,
        venue=info.get("venue", ""),
        city=info.get("city"),
        dates=tuple(info.get("dates", [])),
        toss_winner=toss.get("winner"),
        toss_decision=toss.get("decision"),
        outcome_winner=outcome.get("winner"),
        outcome_by=_format_outcome_by(outcome.get("by")),
    )

    deliveries: list[Delivery] = []
    for innings_index, innings in enumerate(raw.get("innings", []), start=1):
        batting_team = innings["team"]
        bowling_team = next((t for t in teams if t != batting_team), "")
        for over_block in innings.get("overs", []):
            over_num = over_block["over"]
            # Ball number = position in this over's delivery list (1-based), not the
            # fractional part of cricsheet's own "actual_delivery" field -- that field
            # reuses the same ball number for a wide/no-ball *and* its re-bowled ball
            # (both come back as e.g. "17.5"), which would collide as a join key.
            # List order has no such collision and still counts every ball bowled.
            for ball_num, ball in enumerate(over_block.get("deliveries", []), start=1):
                extras = ball.get("extras") or {}
                extras_type = next(iter(extras), None)
                wickets = ball.get("wickets") or []
                wicket = wickets[0] if wickets else {}
                deliveries.append(
                    Delivery(
                        match_id=match_id,
                        innings=innings_index,
                        over=over_num,
                        ball=ball_num,
                        batting_team=batting_team,
                        bowling_team=bowling_team,
                        striker=ball["batter"],
                        non_striker=ball["non_striker"],
                        bowler=ball["bowler"],
                        runs_batter=ball["runs"]["batter"],
                        runs_extras=ball["runs"]["extras"],
                        runs_total=ball["runs"]["total"],
                        extras_type=extras_type,
                        wicket_player_out=wicket.get("player_out"),
                        wicket_kind=wicket.get("kind"),
                        phase=phase_for_over(over_num),
                    )
                )
    return meta, deliveries


def _format_outcome_by(by: Optional[dict]) -> Optional[str]:
    if not by:
        return None
    if "wickets" in by:
        return f"{by['wickets']} wickets"
    if "runs" in by:
        return f"{by['runs']} runs"
    return str(by)
