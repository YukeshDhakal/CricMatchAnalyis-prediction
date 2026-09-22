"""Verifying a Supabase end user here, so a browser can call this API without a proxy.

**Why this moved.** Until now the only browser-facing caller was the Next.js app on
Vercel, which held a static `X-API-Key` server-side, checked the user's Supabase session
itself, and forwarded the upload. That worked for tiny clips and could not be made to
work for real ones: a Vercel Serverless Function has a hard request-body ceiling at the
*platform gateway*, before any route handler runs, so a 4.66MB clip came back
`413 FUNCTION_PAYLOAD_TOO_LARGE` with the app's own code never executing. No amount of
configuration fixes the shape of that -- routing raw video bytes through a serverless
function is the wrong architecture for a product whose input is match footage. The
browser now uploads straight to this container instead, which means this service has to
be able to answer "who is this" on its own. That is what this module is.

**What is verified, and against what.** The caller presents their ordinary Supabase
session access token as `Authorization: Bearer <token>`. Two calls to Supabase follow:

1. `GET /auth/v1/user` -- resolves the token to a user. This is the step that makes the
   token *verified* rather than merely *parsed*: it is checked by the Auth server, so an
   expired, revoked or forged token cannot get past it. The `apikey` header is not
   optional here; without it Supabase answers "No API key found in request" and never
   looks at the bearer at all.
2. `GET /rest/v1/video_pipeline_access?select=role&user_id=eq.<id>` -- reads the role,
   sent with the *caller's own* token rather than a privileged one, so PostgREST
   evaluates the table's RLS policy (`user_id = auth.uid()`, SELECT only, no write
   policy at all) as that user. A caller can only ever see their own row, and no
   service-role key is needed or held by this service.

Step 1 cannot be skipped by "just doing the RLS query". Verified against the live
project: the REST query with an unauthenticated bearer returns `200 []`, not an error --
an empty list, which is indistinguishable from a real user who holds no row. It is only
an authenticator because step 1 ran first.

**Why raw HTTP and not supabase-py.** This service's dependency list is already a
multi-GB ML stack; two GETs do not justify another client library and its own transitive
tree in the image. `requests` is already a pinned dependency. The Next.js side used
`@supabase/supabase-js` for the same two calls -- this is deliberately the same sequence,
in the same order, with the same deny-by-default endings, just written out.

**Why the env vars here are not secrets.** `SUPABASE_URL` and `SUPABASE_ANON_KEY` are the
same two public values the web app ships to every browser as `NEXT_PUBLIC_*`. The anon
key grants nothing on its own -- it is a project identifier that RLS then constrains.
The value worth keeping out of the browser is the *static* `THIRD_UMPIRE_API_KEY`, and
this module exists precisely so the browser never needs it.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from typing import Literal

import requests

PipelineRole = Literal["admin", "demo"]

_VALID_ROLES = ("admin", "demo")
_TIMEOUT_SECONDS = 10

# A live run is polled every few seconds for minutes, and each poll would otherwise cost
# two round trips to Supabase. Successful resolutions are cached briefly, keyed by a hash
# of the token rather than the token itself so the process never holds a pile of usable
# credentials in memory.
#
# The revocation window this opens is not actually new: a Supabase access token stays
# cryptographically valid until it expires (signing out invalidates the *refresh* token,
# not the access token already issued), so a token holder had that window regardless.
# What the cache genuinely delays is a role change -- granting or revoking a row in
# video_pipeline_access takes up to this long to take effect. Failures are never cached,
# so a newly-granted user is not locked out for a minute by their own first 403.
_CACHE_TTL_SECONDS = 60
_CACHE_MAX_ENTRIES = 256
_cache: dict[str, tuple[float, "SupabaseCaller"]] = {}
_cache_lock = threading.Lock()


class SupabaseAuthError(Exception):
    """A caller this service will not serve, with an HTTP status and a message meant to
    be shown to the person who sent the request."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class SupabaseCaller:
    user_id: str
    email: str | None
    role: PipelineRole


def _config() -> tuple[str | None, str | None]:
    """Read at call time, not at import.

    `api.server` reads its own env at import and its tests have to `importlib.reload` it
    to change anything. Reading here instead means a test (or a host that sets the vars
    late) needs no reload dance, and costs one dict lookup per request.
    """
    url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    key = (os.environ.get("SUPABASE_ANON_KEY") or "").strip()
    return (url or None, key or None)


def is_configured() -> bool:
    """Whether bearer-token auth is available at all on this deployment. Used to decide
    which "you're not authorized" message a credential-less request deserves."""
    url, key = _config()
    return bool(url and key)


def wants_frames(role: PipelineRole) -> bool:
    """Whether this role gets the heavy half of the job result -- per-track rows and real
    sampled frames with detections and pose keypoints drawn on them.

    Deliberately asymmetric, and deliberately one named function rather than inline at
    the call site: `demo` is the showcase account and gets the fuller picture, while
    `admin` is the owner's own account and stays lighter and faster for quick checks. It
    reads backwards if you assume admin means "more", so it is written down here once.

    This is the rule that used to live in the Next.js proxy (`lib/pipeline-auth.ts`'s
    `wantsFrames`). It moved here with the rest of the decision, not in addition to it --
    there is one copy, on the side that acts on it.
    """
    return role == "demo"


def _cache_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _cache_get(token: str) -> SupabaseCaller | None:
    key = _cache_key(token)
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit is None:
            return None
        expires_at, caller = hit
        if expires_at <= now:
            _cache.pop(key, None)
            return None
        return caller


def _cache_put(token: str, caller: SupabaseCaller) -> None:
    now = time.monotonic()
    with _cache_lock:
        # Prune expired entries first; only if that leaves it still full does anything
        # live get evicted. Bounded so a stream of distinct tokens cannot grow it without
        # limit.
        for key, (expires_at, _) in list(_cache.items()):
            if expires_at <= now:
                _cache.pop(key, None)
        if len(_cache) >= _CACHE_MAX_ENTRIES:
            _cache.pop(next(iter(_cache)), None)
        _cache[_cache_key(token)] = (now + _CACHE_TTL_SECONDS, caller)


def clear_cache() -> None:
    """Drop every cached resolution. For tests, and for anything that needs a role change
    to take effect immediately rather than within the TTL."""
    with _cache_lock:
        _cache.clear()


def resolve_caller(token: str) -> SupabaseCaller:
    """Resolve a Supabase access token to a user who is allowed to run the pipeline.

    Raises `SupabaseAuthError` for every outcome that is not a granted user. There is no
    "unknown" return -- every path either produces a caller with a real role or refuses.
    """
    url, anon_key = _config()
    if not url or not anon_key:
        raise SupabaseAuthError(
            503,
            "Sign-in isn't configured on this server (SUPABASE_URL / SUPABASE_ANON_KEY "
            "are unset), so a bearer token can't be verified. Use an X-API-Key instead, "
            "or set those two variables.",
        )

    cached = _cache_get(token)
    if cached is not None:
        return cached

    headers = {"Authorization": f"Bearer {token}", "apikey": anon_key}

    try:
        user_res = requests.get(f"{url}/auth/v1/user", headers=headers, timeout=_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise SupabaseAuthError(
            503, f"Couldn't reach Supabase to verify your session: {exc}"
        ) from exc

    if user_res.status_code != 200:
        # Supabase answers a malformed or expired token with 403 (`error_code: bad_jwt`),
        # not 401 -- verified against the live project. Forwarding that verbatim would
        # tell the browser "forbidden", which reads as "your account lacks permission"
        # and is the wrong instruction: the fix is to sign in again. Normalised to 401 so
        # 401 means "your token is no good" and 403 means "your account isn't allowed",
        # which is the distinction the UI actually branches on.
        raise SupabaseAuthError(401, "That session has expired or isn't valid. Sign in again.")

    try:
        user = user_res.json()
    except ValueError as exc:
        raise SupabaseAuthError(502, "Supabase returned a response that wasn't JSON.") from exc

    user_id = user.get("id") if isinstance(user, dict) else None
    if not user_id:
        raise SupabaseAuthError(401, "That session has expired or isn't valid. Sign in again.")
    email = user.get("email") if isinstance(user, dict) else None

    try:
        role_res = requests.get(
            f"{url}/rest/v1/video_pipeline_access",
            params={"select": "role", "user_id": f"eq.{user_id}"},
            headers=headers,
            timeout=_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise SupabaseAuthError(
            503, f"Couldn't reach Supabase to read your access level: {exc}"
        ) from exc

    if role_res.status_code != 200:
        raise SupabaseAuthError(
            502, f"Couldn't read your access level (Supabase returned {role_res.status_code})."
        )

    try:
        rows = role_res.json()
    except ValueError as exc:
        raise SupabaseAuthError(502, "Supabase returned a response that wasn't JSON.") from exc

    # No row is the normal state for any account that hasn't been granted live-run
    # access, and under RLS it is indistinguishable from "a row exists but isn't yours"
    # -- which is the point. Either way the answer is the same and says nothing about
    # other accounts. More than one row cannot happen (user_id is the primary key) and is
    # refused rather than guessed at.
    role = None
    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
        role = rows[0].get("role")
    if role not in _VALID_ROLES:
        raise SupabaseAuthError(403, "This account isn't allowed to trigger live pipeline runs.")

    caller = SupabaseCaller(user_id=str(user_id), email=email, role=role)  # type: ignore[arg-type]
    _cache_put(token, caller)
    return caller
