"""Generic direct-URL downloader for the manual intake folders (data/uploads/*)."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve


def download_url(url: str, dest_dir: Path | str, filename: str | None = None) -> Path:
    """Downloads `url` into `dest_dir`, using the URL's own filename unless one is given."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = filename or Path(urlparse(url).path).name or "download"
    dest_path = dest_dir / name
    urlretrieve(url, dest_path)
    return dest_path
