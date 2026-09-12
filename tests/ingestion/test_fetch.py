from pathlib import Path

from ingestion.fetch import download_url


def test_download_url_saves_into_dest_dir(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("hello", encoding="utf-8")
    dest_dir = tmp_path / "dest"

    result = download_url(source.as_uri(), dest_dir)

    assert result == dest_dir / "source.txt"
    assert result.read_text(encoding="utf-8") == "hello"


def test_download_url_honours_explicit_filename(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("hi", encoding="utf-8")
    dest_dir = tmp_path / "dest"

    result = download_url(source.as_uri(), dest_dir, filename="renamed.mp4")

    assert result.name == "renamed.mp4"
