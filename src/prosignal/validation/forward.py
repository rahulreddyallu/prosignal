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
    "Registration", "Progress", "REGISTRATION_NAME", "fingerprint_scheme",
    "MIN_SESSION_COVERAGE", "COVERAGE_GRACE_SESSIONS",
    "register", "load_registration", "verify", "progress",
    "sessions_in_window",
]

REGISTRATION_NAME = "forward_test.json"

#: Target length. Trading sessions rather than calendar days, because the
#: observation count is what matters and holidays do not produce observations.
TARGET_SESSIONS = 375

#: Monthly observations needed before the attribution is graded. Six factors
#: plus an intercept needs seven parameters; eighteen leaves eleven degrees of
#: freedom, which is the whole reason for the eighteen-month figure.
TARGET_MONTHS = 18

#: Below this share of the sessions the market actually printed, the sample is a
#: SELECTION rather than a period -- the registration names this as one of its
#: four invalidation conditions. It lived in `operations` as a constant nothing
#: read, so the criterion was written into every pre-registration and evaluated
#: by nothing. It belongs here, beside the check that enforces it.
MIN_SESSION_COVERAGE = 0.60

#: Coverage is not meaningful over a handful of days: one missed night out of
#: three is 67% and says nothing, while one out of a hundred is a real gap. The
#: ratio is only judged once the window is long enough for it to mean something.
COVERAGE_GRACE_SESSIONS = 20


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
    #: THE QUESTION THE AUDIT TURNED ON. Added before the window opened, which
    #: is the only time a hypothesis may be added: the first registration had
    #: no benchmark-relative test at all, and neither did any other code path
    #: in this repository, so every economic conclusion was stated against
    #: zero. On the selection period the book returns +1.04% per period against
    #: +5.27% for the equal-weight universe it selects from.
    tertiary: str = ""
    #: Conditions under which the test is abandoned rather than graded.
    invalidation: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    #: THE BENCHMARK-RELATIVE HYPOTHESIS.
    #:
    #: `primary` regresses on factors and `secondary` scores the ranking, and
    #: neither asks whether the book beats the alternative of holding the
    #: universe it selects from. Measured on the selection period the shipped
    #: book returns about 4 points per period LESS than an equal-weight hold of
    #: its own eligible universe, so that is the question, and until this field
    #: existed the forward test could be passed by an engine that loses to
    #: buying everything.
    #:
    #: Optional only so that registrations written before it existed still
    #: load. A NEW registration that omits it is refused by `register`.
    tertiary: str = ""

    def _payload(self, *, legacy: bool) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "started_on": self.started_on,
            "config_version": self.config_version,
            "target_sessions": self.target_sessions,
            "target_months": self.target_months,
            "primary": self.primary,
            "secondary": self.secondary,
            "tertiary": self.tertiary,
            "invalidation": sorted(self.invalidation),
        }
        if not legacy:
            # Inside the hash, so a hypothesis cannot be added, softened or
            # deleted once observations have started landing.
            out["tertiary"] = self.tertiary
        return out

    def fingerprint(self, *, legacy: bool = False) -> str:
        """Hash of everything that must not change. Excludes nothing that
        would let the criteria be edited after the fact.

        ``legacy`` recomputes it the way it was computed before the
        benchmark-relative hypothesis existed. Registrations written under that
        scheme are not tampered with, and saying they are would be exactly the
        kind of untrue message this engine is being audited for -- so `verify`
        checks both and `progress` distinguishes the two cases.
        """
        blob = json.dumps(self._payload(legacy=legacy),
                          sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


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
    #: Sessions the market actually printed inside the window, when the caller
    #: can supply them. `sessions_elapsed` counts the ones that produced a
    #: recorded run; the gap between the two is the coverage question.
    sessions_expected: Optional[int] = None
    #: Distinct model fingerprints seen since the start. config_version
    #: covers parameters.yaml only, so this is the one that notices a code
    #: change or a store that grew under an unchanged config.
    model_fingerprints: List[str] = field(default_factory=list)
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
        covered = ""
        if self.coverage is not None:
            covered = (f" Coverage {self.coverage:.0%} of the "
                       f"{self.sessions_expected} sessions the market printed.")
        return (
            f"{self.sessions_elapsed} of {self.sessions_target} sessions "
            f"({self.fraction:.0%}), {self.months_elapsed} of "
            f"{self.months_target} months, {self.runs_recorded} runs recorded."
            f"{covered} Nothing may be concluded until both targets are met."
        )

    @property
    def coverage(self) -> Optional[float]:
        """Share of printed sessions that produced a recorded run."""
        if not self.sessions_expected:
            return None
        return min(self.sessions_elapsed / self.sessions_expected, 1.0)


def sessions_in_window(sessions, started_on: str,
                       today: Optional[dt.date] = None) -> int:
    """How many sessions the market printed between registration and today.

    The denominator for the coverage criterion. Takes the store's session list
    rather than a store, so this module stays free of the data layer and both
    callers -- the API and the CLI -- get the same arithmetic.
    """
    try:
        start = dt.date.fromisoformat(str(started_on)[:10])
    except (TypeError, ValueError):
        return 0
    end = today or dt.date.today()
    n = 0
    for day in sessions or ():
        if isinstance(day, dt.datetime):
            day = day.date()
        if not isinstance(day, dt.date):
            try:
                day = dt.date.fromisoformat(str(day)[:10])
            except ValueError:
                continue
        if start <= day <= end:
            n += 1
    return n


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
    cfg: Any = None,
    unchecked_reason: str = "",
) -> Registration:
    """Write the pre-registration. Refuses to overwrite an existing one.

    Overwriting would silently reset the clock and let a disappointing window
    be replaced by a fresh one, which is the specific failure this whole
    exercise exists to avoid.

    R1 -- AND IT ALSO REFUSES TO OPEN ONTO AN ENGINE THAT IS STILL CHANGING.
    The window this replaces is void because the configuration moved under it.
    Restarting while a restart-blocking finding is open, the data is
    unmanifested, or no epoch describes the engine would void the new window
    the same way, just later and after eighteen months of waiting. So `cfg` is
    gated through `validation.readiness` before anything is written.

    `cfg` has no default that skips the check: a caller must either supply the
    configuration to gate on, or state in `unchecked_reason` why it is not
    gating -- which is legitimate for a fixture and is not legitimate anywhere
    in `src`, where a test asserts no such string appears. An optional gate is
    a gate nobody calls; that is how `holdout.sacred` came to be read by no
    code at all (R12).
    """
    if cfg is None and not unchecked_reason:
        raise ValueError(
            "register() needs the configuration it is registering against, so "
            "it can check that the engine is in a state worth measuring. Pass "
            "cfg=..., or pass unchecked_reason='...' to state why this call "
            "is deliberately ungated."
        )
    if cfg is not None:
        from .readiness import check_may_restart

        check_may_restart(cfg)

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
        tertiary=(
            "TERTIARY. Mean excess return of the paper book over an "
            "equal-weight hold of the eligible universe it selects from, on "
            "the same holding windows, over the 18 forward months. The engine "
            "passes if that excess is positive with an overlap-corrected t of "
            "at least 2.0. The engine is currently expected to FAIL this "
            "test and it is registered for exactly that reason. On the "
            "selection period this test gives mean excess -4.23% per "
            "63-session period, information ratio -0.83, alpha -0.67% and "
            "32.9% of periods beating the benchmark. The book "
            "returns roughly 3.9 points per 63-session period LESS than its "
            "own universe: the ranking earns about +2.0 and the execution "
            "layer -- sizing, the 2.5x ATR stop, the 3R target, the "
            "invalidation exit and costs -- gives back about 5.9. A forward "
            "test whose outcome is not in doubt is not a test; this one's is, "
            "and it is the only hypothesis here that asks whether running the "
            "engine beats not running it."
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
            "The benchmark panel is unavailable for any part of the window. "
            "The tertiary test cannot be evaluated without it, and a book "
            "reported without the alternative it is supposed to beat is the "
            "defect this registration was rewritten to close.",
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


def verify(root: Path, *, allow_legacy: bool = True) -> bool:
    """Whether the registration on disk still matches its own hash.

    ``allow_legacy`` accepts a fingerprint computed before the tertiary
    hypothesis joined the hash. Such a file has not been edited; it was written
    under an earlier definition. It is still not GRADEABLE -- `progress` refuses
    it for having no benchmark-relative hypothesis -- but "you tampered with
    this" and "this predates the current contract" are different statements and
    the engine should not make the first when it means the second.
    """
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
    if stored == reg.fingerprint():
        return True
    return bool(allow_legacy and stored == reg.fingerprint(legacy=True))


def fingerprint_scheme(root: Path) -> str:
    """"current", "legacy", "mismatch", or "absent"."""
    path = _path(root)
    if not path.is_file():
        return "absent"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "mismatch"
    stored = data.pop("fingerprint", None)
    try:
        reg = Registration(**data)
    except TypeError:
        return "mismatch"
    if stored == reg.fingerprint():
        return "current"
    if stored == reg.fingerprint(legacy=True):
        return "legacy"
    return "mismatch"


def progress(
    root: Path,
    ledger_rows,
    *,
    today: Optional[dt.date] = None,
    live_config_version: Optional[str] = None,
    sessions_printed: Optional[int] = None,
) -> Optional[Progress]:
    """Read the ledger and report how far the window has run.

    Deliberately reports NOTHING about performance. Interim results invite
    optional stopping, and a test stopped when it looks good has no p-value
    worth quoting.

    ``live_config_version`` is the configuration the engine is running NOW.
    Without it, drift is detectable only from the versions stamped on ledger
    rows INSIDE the window -- so a config changed between registration and the
    first observation left `broken` empty and the summary reading "Registered
    today; no forward session yet", which is the reassuring wording for a
    window that is already void. Observed exactly that: registered against
    a30a8d48, engine running 9776e5d6, nothing flagged.

    ``sessions_printed`` is how many sessions the MARKET produced inside the
    window. It is the denominator the registration's fourth invalidation
    condition needs and the caller is the only thing that knows it, because
    this module has no store.
    """
    reg = load_registration(root)
    if reg is None:
        return None
    today = today or dt.date.today()
    start = dt.date.fromisoformat(reg.started_on)

    dates: List[dt.date] = []
    versions: set = set()
    prints: set = set()
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
        fp = row.get("model_fingerprint")
        if fp and fp != "unknown/?":
            prints.add(str(fp))

    unique_days = sorted(set(dates))
    months = (today.year - start.year) * 12 + (today.month - start.month)

    broken: List[str] = []
    scheme = fingerprint_scheme(root)
    if scheme == "mismatch":
        broken.append("the pre-registration file no longer matches its hash")
    elif scheme == "legacy":
        broken.append(
            "the pre-registration predates the benchmark-relative hypothesis "
            "and was written under the earlier fingerprint -- it has not been "
            "edited, but it cannot be graded under the current contract")
    if not str(reg.tertiary or "").strip():
        # A registration carrying only `primary` and `secondary` can be passed
        # by an engine that loses to holding its own universe, because nothing
        # in it compares the two. Such a window is not graded silently.
        broken.append(
            "the registration carries no benchmark-relative hypothesis, so "
            "passing it would say nothing about whether running the engine "
            "beats holding the universe it selects from -- re-register"
        )
    # The check that does not need an observation to have landed yet.
    if live_config_version and live_config_version != reg.config_version:
        broken.append(
            f"the engine is running {live_config_version} but the test was "
            f"registered against {reg.config_version} -- the configuration "
            f"changed after registration, so this window is not one experiment"
        )
    # The check config_version cannot make. Same knobs, different model.
    if len(prints) > 1:
        broken.append(
            f"{len(prints)} model fingerprints recorded since the start "
            f"({', '.join(sorted(prints))}) -- the source or the training "
            f"depth changed mid-flight even though the config may not have"
        )
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

    # "Fewer than 60% of expected sessions produce a recorded run, which would
    # make the sample a selection rather than a period." The registration has
    # said this since the first window; nothing evaluated it.
    if sessions_printed and sessions_printed >= COVERAGE_GRACE_SESSIONS:
        covered = len(unique_days) / sessions_printed
        if covered < MIN_SESSION_COVERAGE:
            broken.append(
                f"only {len(unique_days)} of the {sessions_printed} sessions "
                f"the market printed produced a recorded run ({covered:.0%}, "
                f"below the {MIN_SESSION_COVERAGE:.0%} the registration "
                f"requires) -- the sample is a selection, not a period"
            )

    return Progress(
        started_on=reg.started_on,
        latest_session=unique_days[-1].isoformat() if unique_days else None,
        sessions_elapsed=len(unique_days),
        sessions_target=reg.target_sessions,
        months_elapsed=max(months, 0),
        months_target=reg.target_months,
        runs_recorded=len(dates),
        sessions_expected=(int(sessions_printed) if sessions_printed else None),
        config_versions=sorted(versions),
        model_fingerprints=sorted(prints),
        broken=broken,
    )
