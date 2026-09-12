import json
import zipfile

from ingestion.sources.cricsheet import CricsheetSource


def _make_source_zip(zip_path, match_json: dict, filename: str = "111.json"):
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(filename, json.dumps(match_json))


MINIMAL_MATCH = {
    "info": {"teams": ["A", "B"], "match_type": "T20", "gender": "male"},
    "innings": [],
}


def test_first_fetch_downloads_and_records_last_modified(tmp_path, monkeypatch):
    source = CricsheetSource("comp", cache_dir=tmp_path)
    monkeypatch.setattr(source, "_remote_last_modified", lambda url: "Wed, 01 Jan 2026 00:00:00 GMT")

    def fake_urlretrieve(url, dest):
        _make_source_zip(dest, MINIMAL_MATCH)

    monkeypatch.setattr("ingestion.sources.cricsheet.urlretrieve", fake_urlretrieve)

    result = source.fetch()

    assert result == source.extract_dir
    assert (source.extract_dir / "111.json").exists()
    assert source._read_cached_last_modified() == "Wed, 01 Jan 2026 00:00:00 GMT"


def test_second_fetch_skips_download_when_source_unchanged(tmp_path, monkeypatch):
    source = CricsheetSource("comp", cache_dir=tmp_path)
    monkeypatch.setattr(source, "_remote_last_modified", lambda url: "Wed, 01 Jan 2026 00:00:00 GMT")

    calls = []
    monkeypatch.setattr(
        "ingestion.sources.cricsheet.urlretrieve",
        lambda url, dest: (calls.append(url), _make_source_zip(dest, MINIMAL_MATCH))[-1],
    )

    source.fetch()
    assert len(calls) == 1

    source.fetch()  # same Last-Modified as before -- must not re-download
    assert len(calls) == 1


def test_fetch_redownloads_when_source_has_changed(tmp_path, monkeypatch):
    source = CricsheetSource("comp", cache_dir=tmp_path)
    last_modified = ["Wed, 01 Jan 2026 00:00:00 GMT"]
    monkeypatch.setattr(source, "_remote_last_modified", lambda url: last_modified[0])

    calls = []

    def fake_urlretrieve(url, dest):
        calls.append(url)
        _make_source_zip(dest, MINIMAL_MATCH, filename=f"match_{len(calls)}.json")

    monkeypatch.setattr("ingestion.sources.cricsheet.urlretrieve", fake_urlretrieve)

    source.fetch()
    assert len(calls) == 1
    assert (source.extract_dir / "match_1.json").exists()

    last_modified[0] = "Thu, 02 Jan 2026 00:00:00 GMT"  # cricsheet.org updated the archive
    source.fetch()

    assert len(calls) == 2
    assert not (source.extract_dir / "match_1.json").exists()  # stale extract wiped, not merged
    assert (source.extract_dir / "match_2.json").exists()


def test_iter_matches_always_rechecks_freshness_even_when_already_cached(tmp_path, monkeypatch):
    source = CricsheetSource("comp", cache_dir=tmp_path)
    monkeypatch.setattr(source, "_remote_last_modified", lambda url: "Wed, 01 Jan 2026 00:00:00 GMT")
    monkeypatch.setattr(
        "ingestion.sources.cricsheet.urlretrieve", lambda url, dest: _make_source_zip(dest, MINIMAL_MATCH)
    )

    fetch_calls = []
    real_fetch = source.fetch

    def counting_fetch(*a, **kw):
        fetch_calls.append(1)
        return real_fetch(*a, **kw)

    monkeypatch.setattr(source, "fetch", counting_fetch)

    list(source.iter_matches())
    list(source.iter_matches())

    assert len(fetch_calls) == 2  # every call re-checks the live source, cache hit or not
