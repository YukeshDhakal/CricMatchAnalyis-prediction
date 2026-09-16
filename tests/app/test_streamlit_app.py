"""Smoke tests for the live test console, using Streamlit's official headless
AppTest harness. Kept deliberately light: AppTest runs the script in "bare
mode" (no real ScriptRunContext, session state doesn't function) which is a
meaningfully different execution path from a real `streamlit run` session --
these tests catch "the script raises" and "real data is present," not exact
numeric output, which is verified directly against player_reports.stats
instead (tests/player_reports/).

Since the design pass (see app/streamlit_app.py's THEME_CSS), the warehouse
counts on Overview render as raw HTML stat tiles via st.markdown, not
st.metric -- AppTest can't introspect a value out of arbitrary HTML the way
it can `.metric[i].value`, so that assertion reads the numbers back out of
the rendered markup instead.
"""
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
APP_PATH = REPO_ROOT / "app" / "streamlit_app.py"
WAREHOUSE = REPO_ROOT / "data" / "warehouse" / "matches.parquet"

pytestmark = pytest.mark.skipif(
    not WAREHOUSE.exists(), reason="No local warehouse -- run `python -m ingestion.cli ingest-stats` first"
)


def test_app_loads_without_exceptions():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    assert not at.exception
    assert len(at.tabs) == 4


def test_overview_tab_shows_real_counts():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    stat_row = next(
        m.value for m in at.tabs[0].markdown if "tu-stat-row" in m.value
    )
    values = re.findall(r'tu-stat-value tu-mono">([^<]+)<', stat_row)
    assert len(values) == 4  # Matches, Deliveries, Competitions, Players seen
    assert int(values[0].replace(",", "")) > 0  # matches
    assert int(values[1].replace(",", "")) > 0  # deliveries


def test_player_tab_renders_a_report_without_exceptions():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    player_tab = at.tabs[1]
    first_player = player_tab.selectbox[0].options[0]

    player_tab.selectbox[0].select(first_player).run(timeout=60)

    assert not at.exception
    assert len(player_tab.table) == 2  # batting + bowling


def test_rating_tab_computes_without_a_live_llm():
    """The default note-writer radio option is "Template only" -- this must never
    need a running Ollama server or the network to produce a result."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    rating_tab = at.tabs[3]
    assert rating_tab.radio[0].value.startswith("Template only")

    rating_tab.button[0].click().run(timeout=120)

    assert not at.exception
