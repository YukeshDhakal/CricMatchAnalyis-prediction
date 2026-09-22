"""Tests the URL-fetch guard that stands between a public endpoint and `urllib`.

Two halves, deliberately separated:

* **The guard** (`validate_video_url`) is tested against literal addresses, so no test
  here needs a DNS server or an internet connection to prove that loopback, link-local
  and RFC1918 addresses are refused.
* **The download loop** (`fetch_video_url`) is tested against a real HTTP server on
  127.0.0.1 -- real sockets, real chunking, real redirects -- with the host check
  neutralised for those tests only, because the guard's whole job is to refuse the
  loopback address the test server necessarily lives on. Stubbing the server instead
  would leave the size cap and the redirect handler untested against anything real.
"""
from __future__ import annotations

import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from api.video_source import VideoUrlError, fetch_video_url, validate_video_url


# --- the guard ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "file://C:/Windows/win.ini",
        "ftp://example.com/clip.mp4",
        "gopher://example.com/clip.mp4",
        "/etc/passwd",
    ],
)
def test_non_http_schemes_are_refused(url):
    with pytest.raises(VideoUrlError, match="http"):
        validate_video_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://example.com/",
        "https://example.com/clip.html",
        "https://example.com/clip",
        "https://example.com/clip.exe",
    ],
)
def test_urls_that_do_not_point_at_a_video_file_are_refused(url):
    with pytest.raises(VideoUrlError, match="direct"):
        validate_video_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/clip.mp4",
        "http://localhost/clip.mp4",
        "http://169.254.169.254/latest/meta-data.mp4",  # cloud instance metadata
        "http://10.0.0.5/clip.mp4",
        "http://192.168.1.10/clip.mp4",
        "http://172.16.0.1/clip.mp4",
        "http://[::1]/clip.mp4",
        "http://0.0.0.0/clip.mp4",
    ],
)
def test_private_and_loopback_addresses_are_refused(url):
    """The SSRF case. Each of these is a valid video URL by every other check -- the only
    thing wrong with them is where they point."""
    with pytest.raises(VideoUrlError, match="non-public|resolve"):
        validate_video_url(url)


def test_an_unresolvable_host_is_refused_with_a_readable_reason():
    with pytest.raises(VideoUrlError, match="resolve"):
        validate_video_url("https://no-such-host.invalid/clip.mp4")


@pytest.mark.parametrize("suffix", [".mp4", ".MP4", ".mov", ".avi", ".mkv"])
def test_a_public_direct_video_url_passes(suffix):
    url = f"https://93.184.216.34/clips/over3{suffix}"  # a literal public address: no DNS
    assert validate_video_url(url) == url


# --- the download loop ----------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, payload=b"", redirect_to=None, status=200, **kwargs):
        self._payload = payload
        self._redirect_to = redirect_to
        self._status = status
        super().__init__(*args, **kwargs)

    def do_GET(self):  # noqa: N802 -- BaseHTTPRequestHandler's naming
        if self._redirect_to:
            self.send_response(302)
            self.send_header("Location", self._redirect_to)
            self.end_headers()
            return
        if self._status != 200:
            self.send_response(self._status)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        # Deliberately understated: the cap must be enforced on bytes actually received,
        # not on this header, which a hostile server is free to lie about.
        self.send_header("Content-Length", str(len(self._payload)))
        self.end_headers()
        self.wfile.write(self._payload)

    def log_message(self, *args):  # keep pytest output clean
        pass


@pytest.fixture
def serve():
    servers = []

    def _serve(payload=b"", redirect_to=None, status=200):
        handler = partial(_Handler, payload=payload, redirect_to=redirect_to, status=status)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return f"http://127.0.0.1:{httpd.server_address[1]}"

    yield _serve
    for httpd in servers:
        httpd.shutdown()


@pytest.fixture
def allow_loopback(monkeypatch):
    """Neutralises only the address check, so the rest of the guard (scheme, extension,
    redirect re-validation) is still live in these tests."""
    import api.video_source as module

    monkeypatch.setattr(module, "_assert_public_host", lambda host: None)


def test_a_real_fetch_writes_the_real_bytes(serve, allow_loopback, tmp_path):
    payload = b"\x00\x01video-ish bytes" * 500
    base = serve(payload=payload)
    dest = tmp_path / "clip.mp4"

    fetch_video_url(f"{base}/clip.mp4", dest, max_bytes=1_000_000)

    assert dest.read_bytes() == payload


def test_a_file_over_the_cap_is_refused_and_leaves_no_partial_file(
    serve, allow_loopback, tmp_path
):
    base = serve(payload=b"x" * 200_000)
    dest = tmp_path / "clip.mp4"

    with pytest.raises(VideoUrlError, match="larger than"):
        fetch_video_url(f"{base}/clip.mp4", dest, max_bytes=50_000)

    assert not dest.exists(), "a rejected download left a truncated file behind"


def test_an_empty_response_is_not_treated_as_a_video(serve, allow_loopback, tmp_path):
    base = serve(payload=b"")
    dest = tmp_path / "clip.mp4"
    with pytest.raises(VideoUrlError, match="empty"):
        fetch_video_url(f"{base}/clip.mp4", dest, max_bytes=1_000_000)
    assert not dest.exists()


def test_an_http_error_is_a_readable_message_not_a_traceback(serve, allow_loopback, tmp_path):
    base = serve(status=404)
    dest = tmp_path / "clip.mp4"
    with pytest.raises(VideoUrlError, match="404"):
        fetch_video_url(f"{base}/clip.mp4", dest, max_bytes=1_000_000)


def test_a_redirect_is_revalidated_rather_than_followed_blindly(
    serve, allow_loopback, tmp_path
):
    """The hole a check on only the original URL leaves: a URL that passes every test and
    then 302s somewhere that wouldn't have."""
    base = serve(redirect_to="https://example.com/not-a-video.html")
    dest = tmp_path / "clip.mp4"

    with pytest.raises(VideoUrlError, match="direct"):
        fetch_video_url(f"{base}/clip.mp4", dest, max_bytes=1_000_000)

    assert not dest.exists()


def test_a_redirect_to_a_valid_video_url_is_still_refused_when_it_is_private(
    serve, tmp_path, monkeypatch
):
    """Same as above but with the address check live on the redirect target only -- the
    metadata-endpoint case, which is the one that actually matters on a cloud host."""
    import api.video_source as module

    real_assert = module._assert_public_host

    def only_check_non_loopback(host):
        if host in ("127.0.0.1", "localhost"):
            return
        real_assert(host)

    monkeypatch.setattr(module, "_assert_public_host", only_check_non_loopback)

    base = serve(redirect_to="http://169.254.169.254/latest/meta-data.mp4")
    dest = tmp_path / "clip.mp4"

    with pytest.raises(VideoUrlError, match="non-public"):
        fetch_video_url(f"{base}/clip.mp4", dest, max_bytes=1_000_000)
