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

__all__ = ["TrialRegistry", "Trial", "registry_path"]

FILENAME = "trial_registry.jsonl"


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

    def as_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "key": self.key, "command": self.command, "label": self.label,
            "recorded_at": self.recorded_at}
        if self.score is not None:
            row["score"] = float(self.score)
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
                                         score=score)
                continue
            at[key] = len(out)
            out.append(Trial(key=key, command=str(blob.get("command", "")),
                             label=str(blob.get("label", "")),
                             recorded_at=str(blob.get("recorded_at", "")),
                             score=score))
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

    # -- writing ----------------------------------------------------------
    @staticmethod
    def key_for(command: str, label: str) -> str:
        return hashlib.sha256(f"{command}\x00{label}".encode("utf-8")).hexdigest()[:16]

    def record(self, command: str, labels: Sequence[str],
               scores: Optional[Sequence[Optional[float]]] = None) -> int:
        """Add any configuration not already recorded. Returns how many are new.

        Idempotent by (command, label): re-running the same comparison does not
        inflate the count. A researcher who reruns `research estimator` twenty
        times has still only looked at those arms once.

        ``scores``, when given, must line up with ``labels`` -- one
        out-of-sample score per configuration, in the same order. A mismatch
        raises rather than being zipped short: a registry whose scores silently
        belong to the wrong arms is worse than one with no scores, because the
        variance computed from it looks like evidence.
        """
        if scores is not None and len(scores) != len(labels):
            raise ValueError(
                f"record() got {len(labels)} labels and {len(scores)} scores; "
                f"they must correspond one to one, or scores must be omitted"
            )
        prior = {t.key: t for t in self.load()}
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
                    fresh.append(Trial(key=key, command=command,
                                       label=str(label), recorded_at=now,
                                       score=score))
                continue
            prior[key] = Trial(key=key, command=command, label=str(label),
                               recorded_at=now, score=score)
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
