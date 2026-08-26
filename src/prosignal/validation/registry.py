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

    def as_row(self) -> Dict[str, str]:
        return {"key": self.key, "command": self.command, "label": self.label,
                "recorded_at": self.recorded_at}


class TrialRegistry:
    """Append-only. Reads cheaply, writes idempotently."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    # -- reading ----------------------------------------------------------
    def load(self) -> List[Trial]:
        if not self.path.is_file():
            return []
        out: List[Trial] = []
        seen = set()
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
            if key in seen:
                continue
            seen.add(key)
            out.append(Trial(key=key, command=str(blob.get("command", "")),
                             label=str(blob.get("label", "")),
                             recorded_at=str(blob.get("recorded_at", ""))))
        return out

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

    def record(self, command: str, labels: Sequence[str]) -> int:
        """Add any configuration not already recorded. Returns how many are new.

        Idempotent by (command, label): re-running the same comparison does not
        inflate the count. A researcher who reruns `research estimator` twenty
        times has still only looked at those arms once.
        """
        existing = {t.key for t in self.load()}
        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        fresh = []
        for label in labels:
            key = self.key_for(command, str(label))
            if key in existing:
                continue
            existing.add(key)
            fresh.append(Trial(key=key, command=command, label=str(label),
                               recorded_at=now))
        if not fresh:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for t in fresh:
                fh.write(json.dumps(t.as_row()) + "\n")
        log.info("trials recorded", extra={"command": command, "new": len(fresh),
                                           "total": len(existing)})
        return len(fresh)

    def effective_trials(self, carried: int = 0) -> int:
        """What the DSR should charge for: recorded trials plus prior campaigns.

        `carried` is `validation.search_budget.cumulative_trials_logged`, which
        covers work done before this registry existed. It cannot be reconstructed
        and is not silently assumed to be zero.
        """
        return self.count() + max(int(carried), 0)
