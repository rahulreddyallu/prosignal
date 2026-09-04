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
    "MIN_SESSION_COVERAGE", "COVERAGE_GRACE_SESSIONS", "SCHEME",
    "register", "load_registration", "verify", "progress",
    "sessions_in_window",
]

REGISTRATION_NAME = "forward_test.json"

#: The contract this file is written under. It is INSIDE the fingerprint, so a
#: registration cannot be silently reinterpreted under a later contract than
#: the one it was written to.
#:
#: v1  primary = attribution intercept on SELF-BUILT factors; secondary = pooled
#:     rank IC; tertiary = excess over the equal-weight universe, bolted on
#:     after the fact.
#: v2  primary = long-short SPREAD alpha against an EXTERNAL factor model;
#:     secondary = long-leg excess over the equal-weight eligible universe; plus
#:     a FALSIFICATION set, which is the part an eighteen-month window can
#:     actually decide, and a POWER statement, which says in advance that the
#:     primary cannot be decided in that time.
SCHEME = "v2"

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
    #: load. Empty under scheme v2, where `secondary` carries the
    #: benchmark-relative question instead.
    tertiary: str = ""

    #: Which contract this registration was written under. See `SCHEME`.
    scheme: str = "v1"

    #: WHAT AN EIGHTEEN-MONTH WINDOW CAN ACTUALLY DECIDE.
    #:
    #: The primary hypothesis cannot be settled here and the power statement
    #: says so in advance. What this window CAN do is falsify: a sign that
    #: flips, an ordering that stops being monotone, a cost model that turns
    #: out to be wrong by a factor, a fill that was never achievable, a breadth
    #: that never materialised. Each of these is decidable on far fewer
    #: observations than an alpha estimate, and any one of them failing is
    #: informative on its own.
    #:
    #: Registering only a hypothesis the window cannot decide is how a test
    #: gets read as a verdict on the strategy when it was a verdict on the
    #: sample size.
    falsification: List[str] = field(default_factory=list)

    #: `expected t = IR x sqrt(years)` at the achieved IR over this window,
    #: computed and written BEFORE the first observation. Its purpose is to
    #: stop an early positive stretch being read as confirmation, and to stop
    #: the eventual failure to reach t=2.0 being read as a fact about the
    #: strategy rather than about the horizon.
    power: str = ""

    #: Instruments the hypotheses need that DO NOT EXIST YET. Named rather than
    #: assumed: the primary regresses on an external factor model that Pass 2
    #: builds, and grading it before that exists would mean substituting the
    #: self-built regressors this registration was rewritten to get away from.
    #: While this list is non-empty the corresponding hypothesis is
    #: NOT_TESTABLE -- which is a distinct outcome from a failure, and is never
    #: upgraded to a pass.
    instruments_required: List[str] = field(default_factory=list)

    def _payload(self, *, legacy: bool) -> Dict[str, Any]:
        """The fields that must not change after the first observation.

        ``legacy`` reproduces the v1 payload EXACTLY -- which means WITHOUT
        `tertiary`, because v1 is by definition the scheme from before the
        benchmark-relative hypothesis joined the hash. The previous version of
        this method wrote `tertiary` into both branches, so the two
        fingerprints were byte-identical, `fingerprint(legacy=True)` was dead
        code and `fingerprint_scheme` could never return "legacy". A tamper
        check with an unreachable branch is a tamper check nobody has tested.
        """
        out: Dict[str, Any] = {
            "started_on": self.started_on,
            "config_version": self.config_version,
            "target_sessions": self.target_sessions,
            "target_months": self.target_months,
            "primary": self.primary,
            "secondary": self.secondary,
            "invalidation": sorted(self.invalidation),
        }
        if legacy:
            return out
        # Everything below is inside the hash, so no hypothesis, falsification
        # criterion or power claim can be added, softened or deleted once
        # observations have started landing.
        out["tertiary"] = self.tertiary
        out["scheme"] = self.scheme
        out["falsification"] = sorted(self.falsification)
        out["power"] = self.power
        out["instruments_required"] = sorted(self.instruments_required)
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
        scheme=SCHEME,
        primary=(
            "PRIMARY. Alpha of the LONG-SHORT SPREAD -- top decile minus bottom "
            "decile of the shipped composite, within the point-in-time eligible "
            "universe -- regressed on an EXTERNAL Indian factor model, over the "
            "registered window. The engine passes if that alpha is positive "
            "with an overlap-corrected t of at least 2.0. It fails if the alpha "
            "is not distinguishable from zero. "
            "EXTERNAL IS THE WHOLE POINT: the previous registration regressed "
            "on six long-short factors built from this engine's OWN ranked "
            "columns, which asks an easier and different question -- a "
            "momentum-driven strategy regressed on a momentum factor built from "
            "its own momentum column will show a high R-squared for reasons "
            "that are arithmetic rather than economic. The regressor set is the "
            "IIM Ahmedabad Indian Fama-French + momentum library (survivorship-"
            "corrected, maintained outside this repository), plus two in-house "
            "controls that library lacks: a low-volatility spread and an "
            "illiquidity spread built on this engine's own universe. "
            "THIS HYPOTHESIS IS NOT_TESTABLE UNTIL THAT MODEL EXISTS: see "
            "`instruments_required`. It is registered now, before any "
            "observation, because that is the only time a hypothesis may be "
            "written down; it will be graded when the instrument is built and "
            "not before, and it is never graded against the self-built "
            "regressors as a substitute."
        ),
        secondary=(
            "SECONDARY, and the benchmark-relative question. Mean excess return "
            "of the LONG LEG over an equal-weight hold of the point-in-time "
            "eligible universe it selects from, on the same holding windows. "
            "The engine passes if that excess is positive with an "
            "overlap-corrected t of at least 2.0. "
            "The engine is currently expected to FAIL this and it is registered "
            "for exactly that reason. Re-run on the current store, the shipped "
            "six-name book loses to an equal-weight hold of its own eligible "
            "universe in every window tested, GROSS as well as net -- cost drag "
            "is under one point a year at this cadence, so the deficit is not a "
            "cost problem. A forward test whose outcome is not in doubt is not "
            "a test; this one's is, and it is the hypothesis that asks whether "
            "running the engine beats not running it."
        ),
        instruments_required=[
            "An EXTERNAL Indian factor-return series (IIMA Fama-French + "
            "momentum) ingested with its own provenance, plus in-house "
            "low-volatility and illiquidity spread controls. Pass 2 builds it. "
            "Until it exists the PRIMARY hypothesis is NOT_TESTABLE and must "
            "be reported as such -- never as a pass, and never graded against "
            "`validation.attribution`'s self-built factors instead.",
            "A long-short decile portfolio series (`validation/legs.py`). Pass "
            "2 builds it. The engine has no short leg anywhere in `src/`, so "
            "the PRIMARY hypothesis currently has no series to regress.",
        ],
        power=(
            "POWER, computed before the first observation so that no interim "
            "reading can be mistaken for a verdict. Expected t = IR x "
            "sqrt(years). At the only information ratio this engine has ever "
            "measured out of sample -- 0.78, on the sealed 2012-2017 window -- "
            "the expected t after 18 months is 0.78 x sqrt(1.5) = 0.96, and "
            "t = 2.0 arrives after about 6.5 years. "
            "THEREFORE THE PRIMARY AND SECONDARY HYPOTHESES ARE REGISTERED TO "
            "BE UNDERPOWERED AT THIS HORIZON, and that is stated here rather "
            "than discovered later. Failing to reach t >= 2.0 in eighteen "
            "months is the EXPECTED outcome under a true effect of this size "
            "and is not evidence against the strategy. What this window can "
            "decide is the falsification set below. Any report that quotes the "
            "primary or secondary result without this paragraph beside it is "
            "misreporting the test."
        ),
        falsification=[
            "SIGN OF THE ROLLING RANK IC. The composite's rank IC against the "
            "forward return, rolling. A sustained negative sign falsifies the "
            "ranking outright and needs far fewer observations than an alpha "
            "estimate. Reference: +0.049 (window A) and +0.036 (window B).",
            "DECILE MONOTONICITY. Spearman correlation between decile index "
            "and mean forward excess, computed per date and then averaged -- "
            "never pooled. A composite whose deciles stop being monotone is "
            "ordering noise in the middle of the distribution. Reference: the "
            "generated results of record.",
            "IC DECAY SHAPE. IC measured at 5, 10, 21, 42 and 63 sessions. The "
            "shipped horizon is 63 and the themes are oriented at 42/21/21/10/"
            "10; a decay curve that peaks somewhere else falsifies the holding "
            "period even while the sign holds.",
            "MODELLED VERSUS REALISED COST. Round-trip cost the model "
            "predicted against what the recorded paper fills imply. The impact "
            "coefficient ships UNVALIDATED at 0.10 inside a declared range of "
            "[0.02, 0.50]; if realised cost sits outside the swept band, every "
            "net figure this engine has produced is wrong by a known amount.",
            "MODELLED VERSUS ACHIEVABLE FILL. Whether the assumed next-session "
            "fill was reachable: circuit-limit states, surveillance measures "
            "and thin-book names where the assumed participation was not "
            "available. A fill that was never achievable falsifies the net "
            "figure without saying anything about the signal.",
            "REALISED BREADTH. Names actually held per period against the "
            "book's slots, and entry events actually taken against the cadence. "
            "The engine's t-statistic is bounded by IC x sqrt(breadth); a book "
            "that runs at four names instead of six has less breadth than the "
            "power statement assumed, and the shortfall is measurable "
            "immediately.",
        ],
        invalidation=[
            "config_version changes during the window -- the observations "
            "after the change came from a different model. config_version now "
            "covers the DATA and the TRAINING WINDOW as well as the "
            "parameters, so a store that grows underneath the window breaks it "
            "too, which the previous scheme could not see.",
            "Any factor, gate or parameter is retuned using data from inside "
            "the window.",
            "The shortlist is acted on with real capital, which introduces a "
            "feedback loop the test cannot separate from the signal.",
            "Fewer than 60% of expected sessions produce a recorded run, "
            "which would make the sample a selection rather than a period.",
            "The benchmark panel is unavailable for any part of the window. "
            "The secondary test cannot be evaluated without it, and a book "
            "reported without the alternative it is supposed to beat is the "
            "defect this registration was rewritten to close.",
        ],
        notes=[
            "Eighteen months adds about six non-overlapping 63-session "
            "windows. That is NOT enough to settle either hypothesis and the "
            "power statement says so in advance.",
            "No result may be reported before both targets are met. Reading an "
            "interim result and stopping when it looks good is optional "
            "stopping, and it invalidates the p-value. The falsification set "
            "is the exception and is deliberately monitored throughout: it "
            "tests for a broken instrument, not for a favourable one, and "
            "stopping early on a falsification is the correct response.",
            "Nothing here is funded and nothing here places an order. "
            "docs/EXECUTION_GATE.md governs.",
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
    # THE BENCHMARK-RELATIVE HYPOTHESIS, wherever this scheme keeps it. A
    # registration that cannot compare the book against holding its own
    # universe can be passed by an engine that loses to doing nothing, and such
    # a window is never graded silently.
    if str(reg.scheme or "v1") == "v2":
        if "equal-weight" not in str(reg.secondary or ""):
            broken.append(
                "the v2 registration's secondary hypothesis is not "
                "benchmark-relative, so passing it would say nothing about "
                "whether running the engine beats holding the universe it "
                "selects from -- re-register")
        if not reg.falsification:
            broken.append(
                "the v2 registration carries no falsification set. An "
                "eighteen-month window cannot settle an alpha estimate at this "
                "information ratio, so a registration with nothing falsifiable "
                "in it is registered to produce no evidence at all")
        if not str(reg.power or "").strip():
            broken.append(
                "the v2 registration carries no power statement, so its "
                "eventual failure to reach t >= 2.0 could be read as a fact "
                "about the strategy rather than about the horizon")
    elif not str(reg.tertiary or "").strip():
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
