"""What "today" means to an engine that decides on Indian market sessions.

`runtime.timezone` has been declared in parameters.yaml since v1 with the value
"Asia/Kolkata" and NOTHING READ IT. Every timestamp in the project is naive
local time and every "today" is `dt.date.today()`, which is whatever the host's
clock says. On the deployed instance that happens to be right, because
cloud-init runs `timedatectl set-timezone Asia/Kolkata` once at first boot --
correctness resting on a shell command in a provisioning script that nobody
re-runs.

It matters in exactly one place and it matters a lot there. The staleness gate
counts weekdays between the store's last session and TODAY, against a tolerance
of one. Get "today" wrong by a day and the engine either refuses to run on data
that is current, or runs on data that is a session stale. A UTC host at 20:30
IST is already on the next calendar day, which is the second of those.

So this module answers the question once, from the configured timezone, and
falls back to the host clock when the zone database is unavailable rather than
failing a run over a missing tzdata package.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Optional

__all__ = ["market_timezone", "market_today", "market_now"]

#: Used when the config cannot be read or names a zone this host does not have.
DEFAULT_TIMEZONE = "Asia/Kolkata"


def market_timezone(config: Any = None) -> Optional[str]:
    """The configured zone name, or None when nothing declares one."""
    if config is None:
        return DEFAULT_TIMEZONE
    params = getattr(config, "params", config)
    runtime = getattr(params, "runtime", None)
    name = getattr(runtime, "timezone", None) if runtime is not None else None
    name = getattr(name, "value", name)
    return str(name) if name else DEFAULT_TIMEZONE


def market_now(config: Any = None) -> dt.datetime:
    """Wall-clock time in the market's timezone, as a naive datetime.

    Naive on purpose: every timestamp the engine writes is naive, and returning
    an aware one here would make comparisons against them raise.
    """
    name = market_timezone(config)
    if name:
        try:
            from zoneinfo import ZoneInfo
            return dt.datetime.now(ZoneInfo(name)).replace(tzinfo=None)
        except Exception:          # noqa: BLE001 - missing tzdata is not fatal
            pass
    return dt.datetime.now()


def market_today(config: Any = None) -> dt.date:
    """The market's current date. The one the staleness gate must use."""
    return market_now(config).date()
