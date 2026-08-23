"""Access control for a deployed instance.

`api.auth_token` has existed in the configuration since v1 with the comment
"set a token if you ever expose this". Nothing read it. The API had no
middleware, no dependency and no header check, so the token was a note to the
reader rather than a control -- and the endpoints behind it include
`/analysis/run`, which starts a minutes-long job, and `/admin/bootstrap`,
which starts a hours-long one.

On a laptop bound to 127.0.0.1 that gap costs nothing. The moment the process
binds to 0.0.0.0 it is the whole security model, so this module fails CLOSED:
binding publicly without a token is refused at startup rather than served.

The token is read from PROSIGNAL_AUTH_TOKEN in preference to the config file,
because a secret committed to a repository is not a secret and the config file
is tracked.
"""

from __future__ import annotations

import hmac
import os
from typing import Optional

__all__ = ["ENV_VAR", "HOSTED_ENV_MARKERS", "OPEN_PATHS", "assert_safe_to_serve",
           "is_public_bind", "looks_hosted", "resolve_token", "token_matches"]

ENV_VAR = "PROSIGNAL_AUTH_TOKEN"

#: Never require a token for these. Platform health checks cannot carry one,
#: and a health endpoint that 401s reads to the platform as a dead service --
#: which gets the instance restarted in a loop.
#:
#: `/auth` is here because it is the endpoint that CHECKS the token. Guarding
#: it with the middleware makes signing in require being signed in, and the
#: browser has no way to break the cycle: it cannot set an Authorization header
#: on a top-level navigation. The endpoint validates the token itself and
#: returns 401 on a wrong one, so nothing is unguarded.
#: `/` is the interface shell: static HTML and CSS with no data in it. It has
#: to be open, because the sign-in screen lives INSIDE it -- returning 401 for
#: the page means the browser renders raw JSON and the owner of the instance
#: has no way to sign in at all. The shell then fetches its data, those
#: requests 401, and the screen appears. Static shell open, data closed.
OPEN_PATHS = frozenset({"/health", "/ready", "/auth", "/", "/index.html",
                        "/favicon.ico"})

#: The shortest token worth having. Below this an online guess is cheap.
MIN_TOKEN_LENGTH = 24


def resolve_token(config) -> Optional[str]:
    """Environment first, then config. Empty strings are not tokens."""
    from_env = os.environ.get(ENV_VAR, "").strip()
    if from_env:
        return from_env
    configured = getattr(getattr(config.params, "api", None), "auth_token", None)
    if configured is None:
        return None
    configured = str(configured).strip()
    return configured or None


def is_public_bind(host: Optional[str]) -> bool:
    """Whether this host string accepts connections from off-box."""
    if not host:
        return False
    return str(host).strip() not in {"127.0.0.1", "localhost", "::1", ""}


#: Every managed platform sets this, and nothing local does. It is the most
#: reliable signal available inside create_app, which cannot see the --host
#: uvicorn was launched with.
HOSTED_ENV_MARKERS = ("PORT", "RENDER", "FLY_APP_NAME", "RAILWAY_ENVIRONMENT",
                      "DYNO", "K_SERVICE")


def looks_hosted() -> bool:
    return any(os.environ.get(k) for k in HOSTED_ENV_MARKERS)


def assert_safe_to_serve(config, host: Optional[str] = None) -> None:
    """Refuse to start a public, unauthenticated instance.

    Failing closed rather than warning, because a warning in a deploy log is
    read once and a running service is reachable for as long as it is up.
    """
    if not is_public_bind(host) and not looks_hosted():
        return
    token = resolve_token(config)
    if not token:
        where = f"on {host!r}" if host else "in a hosted environment"
        raise RuntimeError(
            f"refusing to serve {where} without an access token. This "
            f"instance exposes /analysis/run and /admin/bootstrap, which start "
            f"long jobs, and it has no other protection. Set {ENV_VAR} to a "
            f"random string of at least {MIN_TOKEN_LENGTH} characters, or bind "
            f"to 127.0.0.1."
        )
    if len(token) < MIN_TOKEN_LENGTH:
        raise RuntimeError(
            f"the access token is {len(token)} characters; at least "
            f"{MIN_TOKEN_LENGTH} are required. Generate one with "
            f"`python -c \"import secrets; print(secrets.token_urlsafe(32))\"`."
        )


def token_matches(supplied: Optional[str], expected: str) -> bool:
    """Constant-time comparison.

    A plain `==` on a secret leaks its length and, in principle, its prefix
    through timing. The cost of doing this properly is one function call.
    """
    if not supplied:
        return False
    return hmac.compare_digest(supplied.strip(), expected.strip())
