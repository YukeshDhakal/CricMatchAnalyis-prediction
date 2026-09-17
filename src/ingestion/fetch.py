"""Generic direct-URL downloader for the manual intake folders (data/uploads/*)."""
from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# `urlretrieve`'s bare request carries Python's default User-Agent
# ("Python-urllib/x.y"), which several real CDNs (Google Cloud Storage buckets,
# w3schools' sample-video host, confirmed by hand against both while wiring this
# into the Streamlit app) reject outright with a 403 -- not because the URL or
# content is wrong, just as basic bot-blocking against the default urllib UA. A
# plain desktop-browser User-Agent is enough to pass; this isn't spoofing a
# specific browser's capabilities, just not identifying as a bare script.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def download_url(url: str, dest_dir: Path | str, filename: str | None = None) -> Path:
    """Downloads `url` into `dest_dir`, using the URL's own filename unless one is given."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = filename or Path(urlparse(url).path).name or "download"
    dest_path = dest_dir / name
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request) as response, open(dest_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    return dest_path
