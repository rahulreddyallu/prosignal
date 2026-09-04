"""An append-only record of every configuration this engine has been tried at.

WHY IT EXISTS. The Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) charges
a result for the number of configurations that were tried before it. That number
was a command-line default of 24 and a config field, `cumulative_trials_logged`,
which shipped at 0 with a comment asking a human to update it after every
campaign. Nobody ever did, and nothing checked.

So the engine's central defence against selection bias was a constant somebody
typed once. Two independent problems with that: it is wrong, and it is wrong in
the direction that flatters the result. A single afternoon of research here
routinely spends dozens of trials -- seven estimator arms, two significance
floors, five shortlist widths, eighteen buy/hold bands, four barrier
calibrations -- and every one of them is a look at the same data.

WHAT COUNTS AS A TRIAL. Any configuration whose out-of-sample score was LOOKED
AT. Not any computation: a run that was never compared to another costs nothing.
A run that was compared, and could have been chosen, is a trial whether or not
it won. Trials are recorded by the research commands themselves rather than
declared, because a number a person maintains by hand is a number that drifts to
whatever makes the result look best.

The registry is append-only and content-addressed by (command, configuration).
Re-running the same comparison does not inflate the count; running a NEW
comparison does.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from ..core.logging import get_logger

log = get_logger(__name__)

__all__ = ["TrialRegistry", "Trial", "registry_path",
           "V10_BUDGET", "V10_TOTAL_BUDGET", "PRE_V10_PASS", "BudgetExceeded"]

FILENAME = "trial_registry.jsonl"


# =============================================================================
# The v10 trial budget
# =============================================================================
#
# WHY A BUDGET AND NOT A COUNTER. The registry already counts trials honestly.
# Counting is not the same as spending deliberately: the Deflated Sharpe charges
# every configuration that was looked at, so a campaign that discovers halfway
# through that it has spent thirty trials has already made whatever ships worse,
# and no later discipline can give them back. On this engine's own numbers the
# DSR reads 0.030 against 4,877 trials. The remaining room is small and it is
# the reason the v10 plan allocates trials per pass in advance instead of
# letting each pass take what it needs.
#
# SO THE REGISTRY REFUSES. A campaign that would take a pass past its allocation
# raises `BudgetExceeded` and records NOTHING -- not the part that fits, either.
# Recording a prefix would leave the researcher holding a half-run comparison
# whose arms are already charged, which is the worst of both.
#
# WHAT A PASS IS. The eight passes of the v10 build plan. The allocation below
# is that document's, unchanged; it sums to V10_TOTAL_BUDGET and a test asserts
# it. P0 and P1 are zero on purpose -- reconciliation and data ingestion look at
# no out-of-sample score, so they cost nothing and any trial charged to them is
# a sign that modelling has leaked into a pass that was meant to have none.
V10_BUDGET: Dict[str, int] = {
    "P0": 0,    # reconcile, re-register, freeze -- no research
    "P1": 0,    # point-in-time evidence expansion -- data, not models
    "P2": 2,    # the long-short measurement layer
    "P3": 12,   # independent information
    "P4": 4,    # expected return, uncertainty, cost
    "P5": 8,    # breadth and portfolio construction
    "P6": 4,    # crash control
    "P7": 6,    # validation and pre-registration
    "P8": 4,    # paper-trading readiness (cost calibration)
}

V10_TOTAL_BUDGET = 40

#: Where trials recorded before the budget existed are attributed. They are
#: real and they are still charged by the DSR through `effective_trials`; what
#: they are NOT is v10 spending, so they get a bucket with no allocation rather
#: than being back-dated into a pass that had not been designed when they ran.
PRE_V10_PASS = "pre-v10"


class BudgetExceeded(RuntimeError):
    """A campaign would take a pass past its pre-registered allocation.

    Carries the arithmetic so the caller can report it rather than re-deriving
    it: the pass, its allocation, what it has already spent, and how many new
    configurations the refused campaign contained.
    """

    def __init__(self, pass_id: str, allocation: int, spent: int,
                 requested: int) -> None:
        self.pass_id = pass_id
        self.allocation = allocation
        self.spent = spent
        self.requested = requested
        self.remaining = max(allocation - spent, 0)
        super().__init__(
            f"trial budget exceeded: pass {pass_id} is allocated {allocation} "
            f"trial(s), has already spent {spent}, and this campaign would add "
            f"{requested} new configuration(s) -- {self.remaining} remain. "
            f"Nothing was recorded. Either narrow the campaign to "
            f"{self.remaining} configuration(s), or re-allocate the budget "
            f"deliberately and say so on the result: every configuration "
            f"looked at is charged by the Deflated Sharpe whether or not it "
            f"was recorded, so spending past the allocation makes whatever "
            f"ships less credible, not more."
        )


def registry_path(curated: Path) -> Path:
    return Path(curated) / FILENAME


@dataclass(frozen=True)
class Trial:
    key: str
    command: str
    label: str
    recorded_at: str
    #: The out-of-sample score this configuration produced, per period, when the
    #: recording command knew it.
    #:
    #: WHY THIS FIELD EXISTS. The Deflated Sharpe needs two inputs: how many
    #: configurations were tried, and how much their Sharpes VARIED. The
    #: registry supplied the first and nothing supplied the second, so every
    #: DSR this engine has ever printed had to guess Var[SR] -- and the answer
    #: is more sensitive to that guess than to anything else it is given. On
    #: this engine's own evidence, moving between two defensible guesses moved
    #: the result from 0.38 FAIL to 0.91. Counting the trials and discarding
    #: what they scored is counting the denominator and throwing away the
    #: numerator.
    score: Optional[float] = None
    #: Which v10 pass spent this trial, or `PRE_V10_PASS` for the trials
    #: recorded before the budget existed. Absent on every row written before
    #: this field did, which is why `load` defaults it rather than requiring it:
    #: a registry that refused to read its own history would be one nobody
    #: could compare a new campaign against.
    pass_id: str = PRE_V10_PASS

    def as_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "key": self.key, "command": self.command, "label": self.label,
            "recorded_at": self.recorded_at}
        if self.score is not None:
            row["score"] = float(self.score)
        if self.pass_id and self.pass_id != PRE_V10_PASS:
            row["pass_id"] = self.pass_id
        return row


class TrialRegistry:
    """Append-only. Reads cheaply, writes idempotently."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    # -- reading ----------------------------------------------------------
    def load(self) -> List[Trial]:
        if not self.path.is_file():
            return []
        out: List[Trial] = []
        at: Dict[str, int] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                blob = json.loads(line)
                key = str(blob["key"])
            except (ValueError, KeyError):
                # A corrupt line is not a reason to under-count. It is skipped
                # for identity but the file is not rewritten or truncated.
                log.warning("unreadable trial-registry line; skipped",
                            extra={"path": str(self.path)})
                continue
            raw = blob.get("score")
            try:
                score = None if raw is None else float(raw)
            except (TypeError, ValueError):
                score = None
            if score is not None and score != score:      # NaN
                score = None

            if key in at:
                # A LATER LINE MAY SUPPLY A MISSING SCORE AND NOTHING ELSE.
                #
                # The trials already on this registry were recorded before a
                # score field existed, and `record` is idempotent by key -- so
                # without this they could never acquire one, and Var[SR] would
                # stay unmeasurable forever on the very campaign it is meant to
                # charge for. Supplementing an absent score does not change the
                # COUNT, which is the property that must stay append-only, and
                # an existing score is never overwritten: a re-run that scored
                # differently is a different measurement, not a correction, and
                # silently replacing the first with the second would let a
                # disappointing arm be re-rolled.
                prior = out[at[key]]
                if prior.score is None and score is not None:
                    out[at[key]] = Trial(key=prior.key, command=prior.command,
                                         label=prior.label,
                                         recorded_at=prior.recorded_at,
                                         score=score, pass_id=prior.pass_id)
                continue
            at[key] = len(out)
            out.append(Trial(key=key, command=str(blob.get("command", "")),
                             label=str(blob.get("label", "")),
                             recorded_at=str(blob.get("recorded_at", "")),
                             score=score,
                             pass_id=str(blob.get("pass_id") or PRE_V10_PASS)))
        return out

    def recorded_scores(self) -> List[float]:
        """Scores of the trials that recorded one. May be shorter than `count`.

        Returned even when incomplete, because a variance over the subset that
        DID record a score is a real estimate of trial dispersion, while the
        alternative -- assuming a unit variance -- is a placeholder. Callers
        that receive fewer than two get the conservative fallback and are told
        so on the result.
        """
        vals = [t.score for t in self.load() if t.score is not None]
        return [float(v) for v in vals if v == v]      # drops NaN

    def scores_recorded(self) -> int:
        return len(self.recorded_scores())

    def count(self) -> int:
        return len(self.load())

    def by_command(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for t in self.load():
            counts[t.command] = counts.get(t.command, 0) + 1
        return counts

    # -- the v10 budget ---------------------------------------------------
    def by_pass(self) -> Dict[str, int]:
        """Trials recorded per v10 pass. Includes the `pre-v10` bucket."""
        counts: Dict[str, int] = {}
        for t in self.load():
            counts[t.pass_id] = counts.get(t.pass_id, 0) + 1
        return counts

    def spent(self, pass_id: str) -> int:
        return self.by_pass().get(str(pass_id), 0)

    def remaining(self, pass_id: str) -> int:
        """Allocation minus spend, floored at zero. Raises on an unknown pass."""
        allocation = self.allocation(pass_id)
        return max(allocation - self.spent(pass_id), 0)

    @staticmethod
    def allocation(pass_id: str) -> int:
        """This pass's v10 allocation.

        An unknown pass is an error rather than an implicit zero or an implicit
        infinity. A typo'd pass id that silently got no budget would refuse
        legitimate work; one that silently got unlimited budget would defeat the
        whole mechanism, and that is the direction mistakes travel in.
        """
        key = str(pass_id)
        if key not in V10_BUDGET:
            raise KeyError(
                f"unknown v10 pass {key!r}. The budget is declared for "
                f"{', '.join(sorted(V10_BUDGET))}; work that belongs to none of "
                f"them is not v10 work and must not be charged to it."
            )
        return V10_BUDGET[key]

    def budget_report(self) -> List[Dict[str, object]]:
        """One row per pass: allocation, spend, remaining. For the CLI."""
        counts = self.by_pass()
        rows: List[Dict[str, object]] = []
        for pid in sorted(V10_BUDGET):
            spent = counts.get(pid, 0)
            rows.append({"pass": pid, "allocated": V10_BUDGET[pid],
                         "spent": spent,
                         "remaining": max(V10_BUDGET[pid] - spent, 0),
                         "over": max(spent - V10_BUDGET[pid], 0)})
        rows.append({"pass": PRE_V10_PASS, "allocated": None,
                     "spent": counts.get(PRE_V10_PASS, 0),
                     "remaining": None, "over": 0})
        return rows

    # -- writing ----------------------------------------------------------
    @staticmethod
    def key_for(command: str, label: str) -> str:
        return hashlib.sha256(f"{command}\x00{label}".encode("utf-8")).hexdigest()[:16]

    def record(self, command: str, labels: Sequence[str],
               scores: Optional[Sequence[Optional[float]]] = None,
               pass_id: str = PRE_V10_PASS) -> int:
        """Add any configuration not already recorded. Returns how many are new.

        Idempotent by (command, label): re-running the same comparison does not
        inflate the count. A researcher who reruns `research estimator` twenty
        times has still only looked at those arms once.

        ``scores``, when given, must line up with ``labels`` -- one
        out-of-sample score per configuration, in the same order. A mismatch
        raises rather than being zipped short: a registry whose scores silently
        belong to the wrong arms is worse than one with no scores, because the
        variance computed from it looks like evidence.

        ``pass_id`` charges the campaign against a v10 pass allocation. It is
        CHECKED BEFORE ANYTHING IS WRITTEN and the check counts only genuinely
        new configurations, so a re-run of a campaign already recorded costs
        nothing and cannot be refused for a budget it has already paid. A
        campaign that would take the pass past its allocation raises
        `BudgetExceeded` and records NOTHING -- not the prefix that would have
        fit. Recording part of a comparison charges the DSR for arms the
        researcher never got to compare.

        The default is `PRE_V10_PASS`, which has no allocation and is never
        refused: existing callers keep working and their trials stay counted by
        `effective_trials`, they simply are not v10 spending. A v10 campaign
        names its pass.
        """
        if scores is not None and len(scores) != len(labels):
            raise ValueError(
                f"record() got {len(labels)} labels and {len(scores)} scores; "
                f"they must correspond one to one, or scores must be omitted"
            )
        prior = {t.key: t for t in self.load()}
        pass_id = str(pass_id or PRE_V10_PASS)

        # THE GATE, and it runs before a single line is written. Only labels
        # this registry has never seen are charged; `prior` is what makes the
        # re-run of an already-paid campaign free.
        if pass_id != PRE_V10_PASS:
            allocation = self.allocation(pass_id)      # raises on a typo'd pass
            wanted = len({self.key_for(command, str(l)) for l in labels}
                         - set(prior))
            already = self.spent(pass_id)
            if already + wanted > allocation:
                raise BudgetExceeded(pass_id, allocation, already, wanted)

        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        fresh = []
        added = 0
        for n, label in enumerate(labels):
            key = self.key_for(command, str(label))
            raw = None if scores is None else scores[n]
            score = None
            if raw is not None:
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    val = float("nan")
                score = val if val == val else None
            seen = prior.get(key)
            if seen is not None:
                # Already counted. Append a supplementary line ONLY to fill in
                # a score it never had; `load` merges it without changing the
                # count, and an arm that already has a score keeps it.
                if seen.score is None and score is not None:
                    # The supplementary line keeps the ORIGINAL pass. This trial
                    # was spent once, by whoever spent it; a later run that
                    # happens to know its score does not re-charge it to a
                    # different pass's budget.
                    fresh.append(Trial(key=key, command=command,
                                       label=str(label), recorded_at=now,
                                       score=score, pass_id=seen.pass_id))
                continue
            prior[key] = Trial(key=key, command=command, label=str(label),
                               recorded_at=now, score=score, pass_id=pass_id)
            fresh.append(prior[key])
            added += 1
        if not fresh:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for t in fresh:
                fh.write(json.dumps(t.as_row()) + "\n")
        log.info("trials recorded",
                 extra={"command": command, "new": added,
                        "scores_backfilled": len(fresh) - added,
                        "total": len(prior)})
        # NEW TRIALS ONLY. A supplementary line that fills in a missing score
        # is not a new configuration and must not read as one -- the return
        # value is what the caller prints as "n configurations compared".
        return added

    def effective_trials(self, carried: int = 0) -> int:
        """What the DSR should charge for: recorded trials plus prior campaigns.

        `carried` is `validation.search_budget.cumulative_trials_logged`, which
        covers work done before this registry existed. It cannot be reconstructed
        and is not silently assumed to be zero.
        """
        return self.count() + max(int(carried), 0)
