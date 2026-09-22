"""Fetching a clip from a URL, safely enough to expose on a public host.

The Streamlit console has had a "Fetch from a URL" source for a while, and it calls
`ingestion.fetch.download_url` directly. That is fine *there*: Streamlit runs on the
developer's own machine, so "this URL is whatever I typed" is the whole threat model.

It is not fine here. `POST /jobs` is reachable from the internet, so a `video_url` form
field is a request to make the *server* issue an HTTP request to an attacker-chosen
address -- classic SSRF. On a container host that means the platform metadata endpoint,
anything else on the internal network, and (because `urllib` speaks more than HTTP)
`file:///etc/passwd`. `download_url` cannot defend against any of that, and it should not
try to: it is a generic fetcher also used for stats files by the CLI, where the caller is
trusted. So the guard lives here, at the boundary that actually has an untrusted caller.

What is enforced, and why each one:

* **Scheme allowlist (http/https).** Closes `file://`, `ftp://` and friends outright.
* **Video extension.** Same check the Streamlit tab makes, for the same reason: a 200
  response is not proof of a video (a YouTube watch URL downloads perfectly good HTML),
  and catching it here produces a clear message instead of an ffprobe traceback three
  layers down.
* **Resolved address is public.** Every address the hostname resolves to is checked, not
  just the first -- a name with one public and one loopback record would otherwise pass.
* **Redirects are re-validated.** urllib follows redirects by default, so a public URL
  that 302s to `http://169.254.169.254/` would sail straight past a check done only on
  the original URL.
* **Byte cap while streaming.** A `Content-Length` header is a claim, not a fact, so the
  cap is enforced on bytes actually written, and the partial file is removed on trip.

Known and accepted limit: this resolves the hostname to validate it and then hands the
URL to urllib, which resolves it again. A DNS entry that changes between those two
lookups (a rebinding attack) would defeat the address check. Closing that properly means
connecting to a pinned, already-validated IP and carrying the original host through TLS
verification -- a real amount of machinery for an endpoint that already requires a
server-held API key to reach at all. Documented rather than silently ignored.
"""
from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
_USER_AGENT = "third-umpire/1.0 (+https://github.com/YukeshDhakal/CricMatchAnalyis-prediction)"
_CHUNK = 256 * 1024


class VideoUrlError(ValueError):
    """A URL this server will not fetch, with a reason meant to be shown to the user."""


def _assert_public_host(host: str | None) -> None:
    if not host:
        raise VideoUrlError("That URL has no hostname.")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise VideoUrlError(f"Could not resolve '{host}': {exc}") from exc

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise VideoUrlError(
                f"'{host}' resolves to a non-public address ({address}). This server only "
                "fetches clips from public hosts."
            )


def validate_video_url(url: str) -> str:
    """Raises `VideoUrlError` unless `url` is a public http(s) URL ending in a video
    extension. Returns the URL unchanged so it can be used inline."""
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise VideoUrlError(f"That isn't a URL this server can parse: {exc}") from exc

    if parsed.scheme not in ALLOWED_SCHEMES:
        raise VideoUrlError(
            f"Only http and https URLs are fetched (got '{parsed.scheme or 'no scheme'}')."
        )
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise VideoUrlError(
            "That URL doesn't point at a video file. It needs to be a *direct* link "
            f"ending in {', '.join(sorted(ALLOWED_VIDEO_SUFFIXES))} -- a page that merely "
            "embeds a player (a YouTube watch link, say) can't be fetched this way."
        )
    _assert_public_host(parsed.hostname)
    return url


class _ValidatingRedirectHandler(HTTPRedirectHandler):
    """Applies the same host check to every hop, not just the URL the caller typed."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        validate_video_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_video_url(url: str, dest_path: Path, max_bytes: int) -> Path:
    """Downloads a validated video URL to `dest_path`, refusing anything over `max_bytes`.

    Raises `VideoUrlError` for anything the caller could fix by supplying a different URL
    (bad scheme, private host, oversized file, dead link) -- the server turns those into a
    4xx with the message attached, rather than a 500 that reads as "this feature broke".
    """
    validate_video_url(url)
    opener = build_opener(_ValidatingRedirectHandler)
    request = Request(url, headers={"User-Agent": _USER_AGENT})

    written = 0
    try:
        with opener.open(request, timeout=30) as response, open(dest_path, "wb") as out:
            while chunk := response.read(_CHUNK):
                written += len(chunk)
                if written > max_bytes:
                    raise VideoUrlError(
                        f"That video is larger than the {max_bytes // (1024 * 1024)}MB limit."
                    )
                out.write(chunk)
    except VideoUrlError:
        dest_path.unlink(missing_ok=True)
        raise
    except HTTPError as exc:
        dest_path.unlink(missing_ok=True)
        raise VideoUrlError(f"That URL returned HTTP {exc.code}.") from exc
    except (URLError, OSError) as exc:
        dest_path.unlink(missing_ok=True)
        raise VideoUrlError(f"Couldn't fetch that URL: {exc}") from exc

    if written == 0:
        dest_path.unlink(missing_ok=True)
        raise VideoUrlError("That URL returned an empty response, not a video.")
    return dest_path
