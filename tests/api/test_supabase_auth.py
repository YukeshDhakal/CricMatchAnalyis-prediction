"""Tests `api.supabase_auth` -- the module that decides whether a browser's Supabase
session may spend this server's CPU.

The two Supabase calls are stubbed rather than made for real. That is deliberate and not
laziness: a test suite that needed live Supabase credentials would be unrunnable in CI
and would fail for reasons that have nothing to do with this repo. What is stubbed is
only the transport; the sequencing, the header construction, the status mapping and
every deny path are the real code.

The stub's *shapes* are not invented. They were taken from the live project
(lshwbwxjlfsjtyzpwuiy) during this pass, which is why a bad token here returns 403 and
not the 401 you might assume -- that is genuinely what Supabase Auth answers, and the
mapping test below exists because getting it wrong would tell a signed-out user they
lack permission instead of telling them to sign in.
"""
from __future__ import annotations

import pytest
import requests

from api import supabase_auth
from api.supabase_auth import SupabaseAuthError

USER_ID = "5a3f52b0-d3ad-4756-99bc-3d99c2e0140d"


class FakeResponse:
    def __init__(self, status_code: int, payload, *, raw: str | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self._raw = raw

    def json(self):
        if self._raw is not None:
            raise ValueError("not json")
        return self._payload


class FakeSupabase:
    """Stands in for the three GETs, recording what it was asked so the test can assert on
    the headers -- the `apikey` header in particular, whose absence makes Supabase ignore
    the bearer token entirely.

    `rows` is the `video_pipeline_access` answer (the hand-issued admin/demo grant) and
    `accounts` is the `tu_accounts` one (the self-serve coach/player product account).
    They default to "an admin, with no product account" -- the shape of the two accounts
    that existed before signup did.
    """

    def __init__(
        self,
        *,
        user=None,
        user_status=200,
        rows=None,
        rows_status=200,
        accounts=None,
        accounts_status=200,
    ) -> None:
        self.user = user if user is not None else {"id": USER_ID, "email": "someone@example.com"}
        self.user_status = user_status
        self.rows = rows if rows is not None else [{"role": "admin"}]
        self.rows_status = rows_status
        self.accounts = accounts if accounts is not None else []
        self.accounts_status = accounts_status
        self.calls: list[tuple[str, dict, dict]] = []

    def __call__(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, dict(headers or {}), dict(params or {})))
        if url.endswith("/auth/v1/user"):
            return FakeResponse(self.user_status, self.user)
        if url.endswith("/rest/v1/video_pipeline_access"):
            return FakeResponse(self.rows_status, self.rows)
        if url.endswith("/rest/v1/tu_accounts"):
            return FakeResponse(self.accounts_status, self.accounts)
        raise AssertionError(f"unexpected URL: {url}")


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    supabase_auth.clear_cache()
    yield
    supabase_auth.clear_cache()


def _install(monkeypatch, fake: FakeSupabase) -> FakeSupabase:
    monkeypatch.setattr(supabase_auth.requests, "get", fake)
    return fake


# --- the happy path -------------------------------------------------------------------


def test_a_valid_token_resolves_to_its_role(monkeypatch):
    fake = _install(monkeypatch, FakeSupabase(rows=[{"role": "demo"}]))
    caller = supabase_auth.resolve_caller("good-token")
    assert caller.user_id == USER_ID
    assert caller.role == "demo"
    assert caller.account_type is None
    assert caller.email == "someone@example.com"


def test_a_product_account_with_no_pipeline_role_is_allowed(monkeypatch):
    """The grant a self-serve signup actually gets. Before this existed, a user who had
    just created a player account was refused by the same 403 a stranger got -- which
    made the signup flow lead nowhere."""
    _install(monkeypatch, FakeSupabase(rows=[], accounts=[{"account_type": "player"}]))
    caller = supabase_auth.resolve_caller("player-token")
    assert caller.role is None
    assert caller.account_type == "player"


def test_holding_both_grants_keeps_both(monkeypatch):
    """They are independent facts and neither is derived from the other, so resolving one
    must not overwrite or imply the other."""
    _install(
        monkeypatch,
        FakeSupabase(rows=[{"role": "demo"}], accounts=[{"account_type": "coach"}]),
    )
    caller = supabase_auth.resolve_caller("both-token")
    assert caller.role == "demo"
    assert caller.account_type == "coach"


def test_a_product_account_does_not_confer_a_pipeline_role(monkeypatch):
    """The escalation this shape has to refuse. `tu_accounts` is self-insertable at
    signup -- that is the whole point of self-serve -- so if a row there implied any
    `video_pipeline_access` role, signing up would be a way to grant yourself the
    showcase account's expensive frame output."""
    _install(monkeypatch, FakeSupabase(rows=[], accounts=[{"account_type": "coach"}]))
    caller = supabase_auth.resolve_caller("coach-token")
    assert caller.role is None
    assert supabase_auth.wants_frames(caller.role) is False


def test_every_call_sends_the_bearer_and_the_apikey_header(monkeypatch):
    """Two separate failure modes, one assertion each.

    Without `apikey`, Supabase answers "No API key found in request" and never looks at
    the token -- so a missing apikey would make *every* sign-in fail. Without the user's
    own bearer on the REST calls, PostgREST would evaluate RLS as the anonymous role and
    both lookups would come back empty -- so a missing bearer would make every sign-in
    fail *closed but wrongly*, as a 403 that looks like a revoked account.
    """
    fake = _install(monkeypatch, FakeSupabase())
    supabase_auth.resolve_caller("good-token")

    assert len(fake.calls) == 3
    for url, headers, _params in fake.calls:
        assert headers["Authorization"] == "Bearer good-token", url
        assert headers["apikey"] == "anon-key", url


def test_both_grant_lookups_are_scoped_to_the_resolved_user(monkeypatch):
    """RLS already scopes them, but the explicit filter is what makes that a belt-and-
    braces arrangement rather than a single point of failure: if either policy were ever
    loosened, this filter still returns one user's row."""
    fake = _install(monkeypatch, FakeSupabase())
    supabase_auth.resolve_caller("good-token")

    _url, _headers, role_params = fake.calls[1]
    assert role_params["user_id"] == f"eq.{USER_ID}"
    assert role_params["select"] == "role"

    _url, _headers, account_params = fake.calls[2]
    assert account_params["user_id"] == f"eq.{USER_ID}"
    assert account_params["select"] == "account_type"


def test_the_user_is_resolved_before_the_role_is_read(monkeypatch):
    """Order is load-bearing. Verified against the live project: the REST query alone
    answers `200 []` for an unauthenticated bearer -- an empty list, not an error -- so
    it can never be the thing that authenticates anybody. /auth/v1/user must come first
    and its failure must stop the sequence."""
    fake = _install(monkeypatch, FakeSupabase(user_status=403, user={"error_code": "bad_jwt"}))
    with pytest.raises(SupabaseAuthError):
        supabase_auth.resolve_caller("forged")
    assert len(fake.calls) == 1
    assert fake.calls[0][0].endswith("/auth/v1/user")


# --- deny paths -----------------------------------------------------------------------


def test_a_rejected_token_becomes_a_401_not_supabases_403(monkeypatch):
    _install(monkeypatch, FakeSupabase(user_status=403, user={"error_code": "bad_jwt"}))
    with pytest.raises(SupabaseAuthError) as exc:
        supabase_auth.resolve_caller("forged")
    assert exc.value.status_code == 401
    assert "sign in again" in exc.value.detail.lower()


def test_a_user_with_neither_grant_is_403(monkeypatch):
    _install(monkeypatch, FakeSupabase(rows=[], accounts=[]))
    with pytest.raises(SupabaseAuthError) as exc:
        supabase_auth.resolve_caller("real-but-ungranted")
    assert exc.value.status_code == 403


def test_the_denial_does_not_say_which_grant_is_missing(monkeypatch):
    """Both tables are invisible to everyone but their owner. Naming one in the refusal
    would be describing the database's contents to someone who cannot read it."""
    _install(monkeypatch, FakeSupabase(rows=[], accounts=[]))
    with pytest.raises(SupabaseAuthError) as exc:
        supabase_auth.resolve_caller("token")
    detail = exc.value.detail.lower()
    assert "tu_accounts" not in detail
    assert "video_pipeline_access" not in detail


@pytest.mark.parametrize("role", ["superuser", "", None, "ADMIN"])
def test_a_role_outside_the_allowed_set_is_403(monkeypatch, role):
    """The database has a CHECK constraint that should make this impossible. It is
    checked here anyway because this service must not depend on a constraint in another
    system staying in place for its authorization to hold -- note "ADMIN" among the
    cases: the comparison is exact, not case-folded."""
    _install(monkeypatch, FakeSupabase(rows=[{"role": role}], accounts=[]))
    with pytest.raises(SupabaseAuthError) as exc:
        supabase_auth.resolve_caller("token")
    assert exc.value.status_code == 403


@pytest.mark.parametrize("account_type", ["owner", "", None, "COACH", "admin"])
def test_an_account_type_outside_the_allowed_set_grants_nothing(monkeypatch, account_type):
    _install(monkeypatch, FakeSupabase(rows=[], accounts=[{"account_type": account_type}]))
    with pytest.raises(SupabaseAuthError) as exc:
        supabase_auth.resolve_caller("token")
    assert exc.value.status_code == 403


def test_more_than_one_row_is_refused_rather_than_guessed(monkeypatch):
    """Guessing would mean picking a privilege level out of an ambiguous answer. Asserted
    for both tables, because both are now read the same way."""
    _install(
        monkeypatch,
        FakeSupabase(
            rows=[{"role": "admin"}, {"role": "demo"}],
            accounts=[{"account_type": "player"}, {"account_type": "coach"}],
        ),
    )
    with pytest.raises(SupabaseAuthError) as exc:
        supabase_auth.resolve_caller("token")
    assert exc.value.status_code == 403


def test_a_user_response_without_an_id_is_401(monkeypatch):
    _install(monkeypatch, FakeSupabase(user={"email": "x@example.com"}))
    with pytest.raises(SupabaseAuthError) as exc:
        supabase_auth.resolve_caller("token")
    assert exc.value.status_code == 401


def test_supabase_being_unreachable_is_a_503_not_an_allow(monkeypatch):
    """The failure mode worth naming: a network blip must never fall through to "well,
    let them in"."""

    def boom(*args, **kwargs):
        raise requests.ConnectionError("dns exploded")

    monkeypatch.setattr(supabase_auth.requests, "get", boom)
    with pytest.raises(SupabaseAuthError) as exc:
        supabase_auth.resolve_caller("token")
    assert exc.value.status_code == 503


def test_a_rest_error_is_a_502_not_an_allow(monkeypatch):
    _install(monkeypatch, FakeSupabase(rows_status=500, rows={}))
    with pytest.raises(SupabaseAuthError) as exc:
        supabase_auth.resolve_caller("token")
    assert exc.value.status_code == 502


def test_an_account_lookup_error_is_a_502_even_when_the_role_lookup_succeeded(monkeypatch):
    """The failure this one guards against is subtler than the last. A holder of a valid
    `admin` role is already authorized by the time the second lookup runs, so it would be
    easy to let its failure pass unnoticed -- and a caller resolved with a silently-empty
    `account_type` is a caller the server is wrong about."""
    _install(
        monkeypatch,
        FakeSupabase(rows=[{"role": "admin"}], accounts_status=500, accounts={}),
    )
    with pytest.raises(SupabaseAuthError) as exc:
        supabase_auth.resolve_caller("token")
    assert exc.value.status_code == 502


def test_unconfigured_supabase_refuses_bearer_auth(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    assert supabase_auth.is_configured() is False
    with pytest.raises(SupabaseAuthError) as exc:
        supabase_auth.resolve_caller("token")
    assert exc.value.status_code == 503


# --- the cache ------------------------------------------------------------------------


def test_a_repeat_call_is_served_from_cache(monkeypatch):
    """A run is polled every few seconds for minutes; without this, each poll costs two
    round trips to Supabase."""
    fake = _install(monkeypatch, FakeSupabase())
    supabase_auth.resolve_caller("token")
    supabase_auth.resolve_caller("token")
    supabase_auth.resolve_caller("token")
    assert len(fake.calls) == 3  # the first resolution only


def test_different_tokens_do_not_share_a_cache_entry(monkeypatch):
    """The bug this guards against is the worst one this module could have: one user's
    resolved identity being handed to the next caller."""
    fake = _install(monkeypatch, FakeSupabase(rows=[{"role": "admin"}]))
    first = supabase_auth.resolve_caller("token-a")

    fake.user = {"id": "other-user", "email": "demo@example.com"}
    fake.rows = [{"role": "demo"}]
    second = supabase_auth.resolve_caller("token-b")

    assert first.role == "admin"
    assert second.role == "demo"
    assert second.user_id == "other-user"


def test_failures_are_not_cached(monkeypatch):
    """So that granting someone a role takes effect on their next attempt, rather than
    locking them out for the length of the TTL because of their own first 403."""
    fake = _install(monkeypatch, FakeSupabase(rows=[]))
    with pytest.raises(SupabaseAuthError):
        supabase_auth.resolve_caller("token")

    fake.rows = [{"role": "demo"}]
    assert supabase_auth.resolve_caller("token").role == "demo"


def test_the_cache_is_bounded(monkeypatch):
    _install(monkeypatch, FakeSupabase())
    for i in range(supabase_auth._CACHE_MAX_ENTRIES + 50):
        supabase_auth.resolve_caller(f"token-{i}")
    assert len(supabase_auth._cache) <= supabase_auth._CACHE_MAX_ENTRIES


def test_the_raw_token_is_never_a_cache_key(monkeypatch):
    """Keys are hashes, so a memory dump or a debugger session doesn't hand over a pile
    of usable credentials."""
    _install(monkeypatch, FakeSupabase())
    supabase_auth.resolve_caller("a-very-distinctive-token")
    assert "a-very-distinctive-token" not in supabase_auth._cache


# --- the role -> frames rule ----------------------------------------------------------


def test_only_demo_gets_frames():
    """Asymmetric on purpose (demo is the showcase account, admin is the owner's own and
    stays fast). Asserted so nobody "fixes" it into admin >= demo later.

    `None` is the case that matters most now: it is every self-serve signup. Frames are
    the expensive half of a response, so the default a new account lands on has to be the
    cheap one, and it has to stay that way by assertion rather than by luck."""
    assert supabase_auth.wants_frames("demo") is True
    assert supabase_auth.wants_frames("admin") is False
    assert supabase_auth.wants_frames(None) is False
