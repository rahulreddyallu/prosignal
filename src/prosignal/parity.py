"""Backtest-live parity.

The validation in the research pack tests the backtest process. It cannot test
the thing that actually breaks in production: whether the live pipeline, on the
day, saw the same data at the same cutoff that the backtest assumes it saw. A
feed that lands late, a constituent list that updated between the two, a
corporate action ingested a day after the fact -- none of these show up in a
purged walk-forward, because the walk-forward reads a store that has already
settled.

Shadow mode runs the full pipeline live and records what it produced. Later, the
same date is re-run from the settled store, and the two are compared. A clean
diff is evidence the backtest describes the live system. A dirty diff names
exactly which stage disagreed, which is the only useful form of the answer -- a
match/no-match boolean tells you nothing about what to fix.

Nothing here is wired to any downstream action. Shadow runs are marked as such
in the ledger and are not readable as recommendations.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["ParityDiff", "ParityReport", "snapshot_run", "compare_snapshots"]

#: Fields compared, in pipeline order. Ordered so the first disagreement is the
#: earliest stage that diverged, which is almost always the cause of the rest.
_COMPARED = (
    ("universe", "the tradeable set"),
    ("eligible", "names passing Stage 3"),
    ("scored", "names the model scored"),
    ("defended", "names Stage 5 argued against"),
    ("triggered", "names with a live entry trigger"),
    ("buys", "names issued as BUY"),
    ("scores", "per-name composite score"),
)


@dataclass
class ParityDiff:
    """One disagreement between a shadow run and its re-run."""

    field: str
    description: str
    only_live: List[str] = field(default_factory=list)
    only_replay: List[str] = field(default_factory=list)
    changed: Dict[str, Any] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not (self.only_live or self.only_replay or self.changed)

    def summary(self) -> str:
        parts = []
        if self.only_live:
            parts.append(f"{len(self.only_live)} only live ({', '.join(self.only_live[:5])})")
        if self.only_replay:
            parts.append(f"{len(self.only_replay)} only on replay ({', '.join(self.only_replay[:5])})")
        if self.changed:
            parts.append(f"{len(self.changed)} changed")
        return f"{self.field}: " + ("; ".join(parts) if parts else "identical")


@dataclass
class ParityReport:
    """The full comparison for one date."""

    as_of: dt.date
    diffs: List[ParityDiff] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return all(d.clean for d in self.diffs)

    @property
    def first_divergence(self) -> Optional[str]:
        """The earliest stage that disagreed. Later stages usually follow."""
        for d in self.diffs:
            if not d.clean:
                return d.field
        return None

    def render(self) -> str:
        head = f"parity {self.as_of}: " + ("CLEAN" if self.clean else "DIVERGED")
        lines = [head]
        if not self.clean:
            lines.append(f"  first divergence at: {self.first_divergence}")
        for d in self.diffs:
            lines.append("  " + d.summary())
        lines.extend("  note: " + n for n in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        return {
            "as_of": self.as_of.isoformat(),
            "clean": self.clean,
            "first_divergence": self.first_divergence,
            "diffs": [
                {"field": d.field, "only_live": d.only_live,
                 "only_replay": d.only_replay, "changed": d.changed}
                for d in self.diffs if not d.clean
            ],
            "notes": list(self.notes),
        }


def snapshot_run(run) -> Dict[str, Any]:
    """Reduce a pipeline run to the fields parity compares.

    Deliberately small. Comparing whole outputs would drown a real divergence in
    timestamps and run ids, and the fields here are the ones a difference in
    could change a decision.
    """
    output = getattr(run, "output", run)
    funnel = getattr(run, "funnel", {}) or {}
    buys = [r.ticker for r in (getattr(output, "recommendations", None) or [])]
    watch = [r.ticker for r in (getattr(output, "watchlist", None) or [])]
    scores = {
        r.ticker: round(float(r.composite_score), 4)
        for r in (getattr(output, "recommendations", None) or [])
        + (getattr(output, "watchlist", None) or [])
    }
    return {
        "as_of": str(getattr(output, "as_of_date", "")),
        "universe": int(funnel.get("universe_considered", 0)),
        "eligible": int(funnel.get("passed_eligibility", 0)),
        "scored": int(funnel.get("scored", 0)),
        "defended": int(funnel.get("survived_defense", 0)),
        "triggered": int(funnel.get("triggered", 0)),
        "buys": sorted(buys),
        "watchlist": sorted(watch),
        "scores": scores,
    }


def _diff_names(field_name, description, live, replay) -> ParityDiff:
    live_set, replay_set = set(live or []), set(replay or [])
    return ParityDiff(
        field=field_name, description=description,
        only_live=sorted(live_set - replay_set),
        only_replay=sorted(replay_set - live_set),
    )


def compare_snapshots(
    live: Dict[str, Any],
    replay: Dict[str, Any],
    score_tolerance: float = 1e-4,
) -> ParityReport:
    """Diff a shadow snapshot against a replay of the same date."""
    as_of_raw = live.get("as_of") or replay.get("as_of") or ""
    try:
        as_of = dt.date.fromisoformat(str(as_of_raw)[:10])
    except ValueError:
        as_of = dt.date.min

    report = ParityReport(as_of=as_of)
    for name, description in _COMPARED:
        left, right = live.get(name), replay.get(name)

        if name == "scores":
            left = left or {}
            right = right or {}
            changed = {
                k: {"live": left[k], "replay": right[k]}
                for k in set(left) & set(right)
                if abs(float(left[k]) - float(right[k])) > score_tolerance
            }
            report.diffs.append(ParityDiff(
                field=name, description=description,
                only_live=sorted(set(left) - set(right)),
                only_replay=sorted(set(right) - set(left)),
                changed=changed,
            ))
            continue

        if isinstance(left, (list, tuple, set)) or isinstance(right, (list, tuple, set)):
            report.diffs.append(_diff_names(name, description, left, right))
            continue

        if left != right:
            report.diffs.append(ParityDiff(
                field=name, description=description,
                changed={"live": left, "replay": right},
            ))
        else:
            report.diffs.append(ParityDiff(field=name, description=description))

    if report.clean:
        report.notes.append(
            "the live run and the replay agree; the backtest describes what the "
            "live pipeline actually did on this date"
        )
    else:
        report.notes.append(
            f"divergence begins at {report.first_divergence}. Later stages "
            f"consume it, so start there rather than at the buys."
        )
    return report
