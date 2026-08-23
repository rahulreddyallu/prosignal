"""The forward test: a pre-registered, unfunded run against future data.

Nine years of history yields roughly thirty independent 63-session windows for
the price factors and eleven for the value factors, and the project has
established that neither shortening the horizon nor extending history can
change that. Fifteen holdout observations against six factors is why the
attribution's alpha estimate swings from +3.67% to -1.01% depending on which
factors enter. The only remaining source of independent observations is time
that has not happened yet.

WHAT MAKES THIS A TEST RATHER THAN A LOG. Three things, and losing any one of
them turns the exercise back into the in-sample fitting it exists to escape.

The criteria are fixed in advance and hashed. A forward test whose success
condition is decided after the data arrives measures nothing, because any
outcome can be made to look like a pass by choosing what to report. The
pre-registration below is written before the first observation and its hash is
recorded; `verify()` refuses to grade the test if it changed.

The model is frozen. Every run during the window must carry the same
config_version. A change mid-flight -- even an improvement -- means the
observations after it were produced by a different model, and the test either
restarts or reports two shorter windows honestly.

Nothing is funded. The point is to observe the engine's opinions without the
feedback loop that live capital creates.

WHAT IT CAN AND CANNOT SETTLE. Eighteen months is about 375 sessions, which is
only six more non-overlapping 63-session windows. That is NOT enough to settle
the ranking question on its own -- it takes the holdout from six independent
windows to roughly twelve.

It is enough for the question that actually matters. The attribution is run on
the paper portfolio's MONTHLY returns rather than on 63-session ranking
windows: eighteen monthly observations against six factors leaves eleven
degrees of freedom, where fifteen observations left eight and the estimate
would not sit still. The forward test is designed to decide the alpha
question, not the IC question.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "Registration", "Progress", "REGISTRATION_NAME",
    "register", "load_registration", "verify", "progress",
]

REGISTRATION_NAME = "forward_test.json"

#: Target length. Trading sessions rather than calendar days, because the
#: observation count is what matters and holidays do not produce observations.
TARGET_SESSIONS = 375

#: Monthly observations needed before the attribution is graded. Six factors
#: plus an intercept needs seven parameters; eighteen leaves eleven degrees of
#: freedom, which is the whole reason for the eighteen-month figure.
TARGET_MONTHS = 18


@dataclass(frozen=True)
class Registration:
    """What will be measured, decided before any of it is observed."""

    started_on: str
    config_version: str
    engine_version: str
    git_commit: str
    target_sessions: int
    target_months: int

    #: The pre-registered hypotheses. Stated as pass conditions so that a
    #: result cannot later be reinterpreted as a success.
    primary: str
    secondary: str
    #: Conditions under which the test is abandoned rather than graded.
    invalidation: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def fingerprint(self) -> str:
        """Hash of everything that must not change. Excludes nothing that
        would let the criteria be edited after the fact."""
        payload = json.dumps({
            "started_on": self.started_on,
            "config_version": self.config_version,
            "target_sessions": self.target_sessions,
            "target_months": self.target_months,
            "primary": self.primary,
            "secondary": self.secondary,
            "invalidation": sorted(self.invalidation),
        }, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class Progress:
    """Where the test has got to, and what it is not yet entitled to say."""

    started_on: str
    latest_session: Optional[str]
    sessions_elapsed: int
    sessions_target: int
    months_elapsed: int
    months_target: int
    runs_recorded: int
    #: Distinct config_versions seen since the start. More than one means the
    #: model moved and the window is not one experiment.
    config_versions: List[str] = field(default_factory=list)
    broken: List[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return (self.sessions_elapsed >= self.sessions_target
                and self.months_elapsed >= self.months_target
                and not self.broken)

    @property
    def fraction(self) -> float:
        if self.sessions_target <= 0:
            return 1.0
        return min(self.sessions_elapsed / self.sessions_target, 1.0)

    def summary(self) -> str:
        if self.broken:
            return ("The forward test is INVALID: " + "; ".join(self.broken)
                    + ". It must be restarted or reported as two shorter "
                      "windows.")
        if self.complete:
            return (f"Complete: {self.sessions_elapsed} sessions and "
                    f"{self.months_elapsed} months since {self.started_on}. "
                    f"The pre-registered tests may now be run.")
        if self.sessions_elapsed == 0:
            return (
                f"Registered {self.started_on}; no forward session yet. "
                f"Observations are counted by MARKET date, so the first one "
                f"arrives with the first session the market prints after "
                f"registration -- a run made against an earlier close is a "
                f"re-score of data the model has already seen."
            )
        return (
            f"{self.sessions_elapsed} of {self.sessions_target} sessions "
            f"({self.fraction:.0%}), {self.months_elapsed} of "
            f"{self.months_target} months, {self.runs_recorded} runs recorded. "
            f"Nothing may be concluded until both targets are met."
        )


def _path(root: Path) -> Path:
    return Path(root) / REGISTRATION_NAME


def register(
    root: Path,
    *,
    config_version: str,
    engine_version: str,
    git_commit: str,
    started_on: Optional[dt.date] = None,
    overwrite: bool = False,
) -> Registration:
    """Write the pre-registration. Refuses to overwrite an existing one.

    Overwriting would silently reset the clock and let a disappointing window
    be replaced by a fresh one, which is the specific failure this whole
    exercise exists to avoid.
    """
    path = _path(root)
    if path.is_file() and not overwrite:
        raise FileExistsError(
            f"a forward test is already registered at {path}. Starting a new "
            f"one discards the observations collected so far; pass "
            f"overwrite=True only if that is genuinely intended."
        )

    reg = Registration(
        started_on=(started_on or dt.date.today()).isoformat(),
        config_version=config_version,
        engine_version=engine_version,
        git_commit=git_commit,
        target_sessions=TARGET_SESSIONS,
        target_months=TARGET_MONTHS,
        primary=(
            "PRIMARY. Regress the paper portfolio's monthly excess return on "
            "the six long-short factors built by validation.attribution, over "
            "the 18 forward months. The engine passes if the intercept is "
            "positive with an overlap-corrected t of at least 2.0. It fails if "
            "the intercept is not distinguishable from zero. On the holdout "
            "this test gave alpha -1.01% at t -0.38 with only 15 observations "
            "against 6 factors; 18 monthly observations leave 11 degrees of "
            "freedom rather than 8."
        ),
        secondary=(
            "SECONDARY. Pooled rank IC of the daily shortlist against the "
            "63-session forward return, over every forward date with a "
            "complete label. The engine passes if the IC is positive with an "
            "overlap-corrected t of at least 2.0. This is expected to pass on "
            "the holdout evidence and is NOT the question at issue -- it is "
            "recorded so that a failure here would be visible rather than "
            "assumed away."
        ),
        invalidation=[
            "config_version changes during the window -- the observations "
            "after the change came from a different model.",
            "Any factor, gate or parameter is retuned using data from inside "
            "the window.",
            "The shortlist is acted on with real capital, which introduces a "
            "feedback loop the test cannot separate from the signal.",
            "Fewer than 60% of expected sessions produce a recorded run, "
            "which would make the sample a selection rather than a period.",
        ],
        notes=[
            "Eighteen months adds about six non-overlapping 63-session "
            "windows. That is NOT enough to settle the ranking question and "
            "the secondary test is reported with that stated.",
            "The primary test uses monthly portfolio returns precisely "
            "because they are numerous enough to leave the attribution with "
            "degrees of freedom.",
            "No result may be reported before both targets are met. Reading "
            "an interim result and stopping when it looks good is optional "
            "stopping, and it invalidates the p-value.",
        ],
    )
    payload = asdict(reg)
    payload["fingerprint"] = reg.fingerprint()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))
    return reg


def load_registration(root: Path) -> Optional[Registration]:
    path = _path(root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    data.pop("fingerprint", None)
    try:
        return Registration(**data)
    except TypeError:
        return None


def verify(root: Path) -> bool:
    """Whether the registration on disk still matches its own hash."""
    path = _path(root)
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    stored = data.pop("fingerprint", None)
    try:
        reg = Registration(**data)
    except TypeError:
        return False
    return stored == reg.fingerprint()


def progress(
    root: Path,
    ledger_rows,
    *,
    today: Optional[dt.date] = None,
) -> Optional[Progress]:
    """Read the ledger and report how far the window has run.

    Deliberately reports NOTHING about performance. Interim results invite
    optional stopping, and a test stopped when it looks good has no p-value
    worth quoting.
    """
    reg = load_registration(root)
    if reg is None:
        return None
    today = today or dt.date.today()
    start = dt.date.fromisoformat(reg.started_on)

    dates: List[dt.date] = []
    versions: set = set()
    for record in ledger_rows:
        row = record if isinstance(record, dict) else dict(
            getattr(record, "__dict__", {}))
        raw = row.get("date")
        if not raw or row.get("error"):
            continue
        try:
            when = (raw if isinstance(raw, dt.date)
                    else dt.date.fromisoformat(str(raw)[:10]))
        except ValueError:
            continue
        # The MARKET date, not the run date. A run made today against last
        # Friday's close is a re-score of data the model has already seen, not
        # a forward observation -- so the window opens with the first session
        # the market produces after registration.
        if when < start:
            continue
        dates.append(when)
        if row.get("config_version"):
            versions.add(str(row["config_version"]))

    unique_days = sorted(set(dates))
    months = (today.year - start.year) * 12 + (today.month - start.month)

    broken: List[str] = []
    if not verify(root):
        broken.append("the pre-registration file no longer matches its hash")
    if len(versions) > 1:
        broken.append(
            f"{len(versions)} configuration versions recorded since the start "
            f"({', '.join(sorted(versions))})"
        )
    elif versions and reg.config_version not in versions:
        broken.append(
            f"runs carry {sorted(versions)[0]} but the test registered "
            f"{reg.config_version}"
        )

    return Progress(
        started_on=reg.started_on,
        latest_session=unique_days[-1].isoformat() if unique_days else None,
        sessions_elapsed=len(unique_days),
        sessions_target=reg.target_sessions,
        months_elapsed=max(months, 0),
        months_target=reg.target_months,
        runs_recorded=len(dates),
        config_versions=sorted(versions),
        broken=broken,
    )
