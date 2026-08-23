"""A deployed instance must not be open.

`api.auth_token` sat in the configuration from v1 carrying the comment "set a
token if you ever expose this", and nothing read it. There was no middleware,
no dependency and no header check anywhere in the API. On a laptop bound to
127.0.0.1 that costs nothing. The endpoints behind it include /analysis/run,
which starts a minutes-long job, and /admin/bootstrap, which starts an
hours-long one, so the moment the process binds publicly the unread token is
the entire security model.

These tests fail closed in both directions: an unauthenticated hosted instance
must refuse to start, and an authenticated one must actually reject.
"""

from __future__ import annotations

import secrets

import pytest
from fastapi.testclient import TestClient

from prosignal.auth import (
    ENV_VAR, MIN_TOKEN_LENGTH, OPEN_PATHS, assert_safe_to_serve,
    is_public_bind, looks_hosted, resolve_token, token_matches,
)


def good_token() -> str:
    return secrets.token_urlsafe(32)


@pytest.fixture
def hosted(monkeypatch):
    monkeypatch.setenv("PORT", "8000")
    return monkeypatch


# ------------------------------------------------------------- bind safety
@pytest.mark.parametrize("host,public", [
    ("127.0.0.1", False), ("localhost", False), ("::1", False),
    ("0.0.0.0", True), ("192.168.1.10", True),
])
def test_public_binds_are_recognised(host, public):
    assert is_public_bind(host) is public


def test_a_hosted_instance_without_a_token_refuses_to_start(hosted):
    from prosignal.api import create_app

    hosted.delenv(ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match="without an access token"):
        create_app()


def test_a_short_token_is_refused(hosted):
    from prosignal.api import create_app

    hosted.setenv(ENV_VAR, "tooshort")
    with pytest.raises(RuntimeError, match=f"at least {MIN_TOKEN_LENGTH}"):
        create_app()


def test_a_local_instance_still_starts_without_a_token(monkeypatch):
    """The laptop workflow must not need a secret."""
    from prosignal.api import create_app

    for marker in ("PORT", "RENDER", "FLY_APP_NAME", "RAILWAY_ENVIRONMENT",
                   "DYNO", "K_SERVICE", ENV_VAR):
        monkeypatch.delenv(marker, raising=False)
    assert looks_hosted() is False
    create_app()


def test_the_refusal_says_what_is_exposed_and_how_to_fix_it(hosted):
    from prosignal.api import create_app

    hosted.delenv(ENV_VAR, raising=False)
    with pytest.raises(RuntimeError) as err:
        create_app()
    text = str(err.value)
    assert "/analysis/run" in text and "/admin/bootstrap" in text
    assert ENV_VAR in text


# ------------------------------------------------------------- enforcement
@pytest.fixture
def client(hosted):
    from prosignal.api import create_app

    token = good_token()
    hosted.setenv(ENV_VAR, token)
    return TestClient(create_app()), token


@pytest.mark.parametrize("path", sorted(OPEN_PATHS - {"/auth"}))
def test_health_probes_are_never_challenged(client, path):
    """A health endpoint that 401s reads to the platform as a dead service and
    gets the instance restarted in a loop."""
    api, _ = client
    assert api.get(path).status_code == 200


@pytest.mark.parametrize("method,path", [
    ("get", "/"), ("get", "/analysis"), ("get", "/config"),
    ("post", "/analysis/run"), ("post", "/admin/bootstrap"),
    ("post", "/admin/release-memory"), ("get", "/history"),
])
def test_everything_else_is_challenged(client, method, path):
    api, _ = client
    assert getattr(api, method)(path).status_code == 401


def test_a_bearer_header_is_accepted(client):
    api, token = client
    assert api.get("/analysis",
                   headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_an_api_key_header_is_accepted(client):
    api, token = client
    assert api.get("/analysis", headers={"X-API-Key": token}).status_code == 200


def test_a_wrong_token_is_rejected(client):
    api, _ = client
    assert api.get("/analysis",
                   headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_the_challenge_names_the_scheme(client):
    api, _ = client
    assert api.get("/analysis").headers.get("WWW-Authenticate") == "Bearer"


# ------------------------------------------------------------------ sign-in
def test_sign_in_is_open_because_it_checks_the_token_itself(client):
    """Guarding /auth with the middleware makes signing in require being
    signed in, and a browser cannot set a header on a top-level navigation."""
    api, token = client
    assert "/auth" in OPEN_PATHS
    assert api.post("/auth", json={"token": token}).status_code == 200
    assert api.post("/auth", json={"token": "nope"}).status_code == 401


def test_the_cookie_unlocks_the_interface(client):
    api, token = client
    assert api.get("/").status_code == 401
    api.post("/auth", json={"token": token})
    assert api.get("/").status_code == 200


def test_the_cookie_is_not_marked_secure_over_plain_http(client):
    """Hardcoding secure=True is right for the deployed instance and silently
    breaks a local HTTP one: the browser accepts the cookie, never sends it
    back, and every request after a successful sign-in returns 401."""
    api, token = client
    header = api.post("/auth", json={"token": token}).headers["set-cookie"]
    assert "HttpOnly" in header
    assert "secure" not in header.lower()


def test_the_cookie_is_marked_secure_behind_a_tls_proxy(client):
    """Platforms terminate TLS and forward plain HTTP with this header."""
    api, token = client
    header = api.post("/auth", json={"token": token},
                      headers={"x-forwarded-proto": "https"}).headers["set-cookie"]
    assert "secure" in header.lower()


# --------------------------------------------------------------- resolution
def test_the_environment_beats_the_config_file(monkeypatch):
    """A secret committed to a tracked file is not a secret."""
    from prosignal.config.loader import load_config

    cfg = load_config()
    monkeypatch.setenv(ENV_VAR, "from-the-environment-and-long-enough")
    assert resolve_token(cfg) == "from-the-environment-and-long-enough"


def test_an_empty_token_is_not_a_token(monkeypatch):
    from prosignal.config.loader import load_config

    monkeypatch.setenv(ENV_VAR, "   ")
    assert resolve_token(load_config()) is None


def test_comparison_is_constant_time():
    """A plain == on a secret leaks its length and prefix through timing."""
    import inspect

    from prosignal import auth

    assert "compare_digest" in inspect.getsource(auth.token_matches)
    assert token_matches("abc", "abc") is True
    assert token_matches("abc", "abd") is False
    assert token_matches(None, "abc") is False
