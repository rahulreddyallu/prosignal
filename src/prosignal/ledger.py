"""Append-only research ledger.

Every run is recorded, signal or not. A ledger holding only the interesting days
is a biased sample, and the trial count it produces feeds the Deflated Sharpe
Ratio directly.

JSONL, one object per line, appended and never rewritten: append-only by
construction, survives a crash mid-write with at most one truncated line, and is
readable without tooling. There is no query load to justify a database.

Writes are fsync'd, so a run cannot report success while its record sits in a
kernel buffer.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .modelprint import model_fingerprint
from .core.contracts import FinalSignalOutput, LedgerRow, RunContext
from .core.errors import LedgerError
from .core.logging import get_logger

__all__ = ["Ledger", "row_from_output"]

log = get_logger(__name__)


class Ledger:
    """Append-only JSONL store of every analysis run."""

    def __init__(self, ledger_dir: Path) -> None:
        self.dir = Path(ledger_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, when: dt.date) -> Path:
        """One file per year. Keeps any single file readable by eye."""
        return self.dir / f"runs-{when.year}.jsonl"

    def append(self, row: LedgerRow) -> Path:
        """Append one row and fsync it.

        Raises
        ------
        LedgerError
            On any write failure. Deliberately fatal: continuing after failing
            to record a run would silently corrupt the trial count.
        """
        path = self._path(row.date)
        payload = row.model_dump(mode="json")
        line = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:
            raise LedgerError(
                f"could not append run {row.run_id} to {path}: {exc}. The run is "
                f"NOT recorded, so its trial must not be counted as evidence.",
                run_id=row.run_id,
                path=str(path),
            ) from exc
        log.info("ledger appended", extra={"run_id": row.run_id, "file": path.name})
        return path

    # -- reading -------------------------------------------------------------
    def read_all(self) -> List[Dict[str, Any]]:
        return list(self.iter_rows())

    def iter_rows(self) -> Iterator[Dict[str, Any]]:
        """Yield every recorded run, oldest file first.

        A truncated final line (crash mid-append) is skipped with a warning
        rather than raising -- one lost record must not make the whole history
        unreadable.
        """
        for path in sorted(self.dir.glob("runs-*.jsonl")):
            with open(path, "r", encoding="utf-8") as fh:
                for n, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        log.warning(
                            "skipping malformed ledger line",
                            extra={"file": path.name, "line": n},
                        )

    def count(self) -> int:
        return sum(1 for _ in self.iter_rows())

    def trial_count(self) -> int:
        """Distinct trial ids recorded. Feeds the DSR multiple-testing penalty."""
        return len({r.get("trial_id") for r in self.iter_rows() if r.get("trial_id")})

    def last_run(self) -> Optional[Dict[str, Any]]:
        last = None
        for row in self.iter_rows():
            last = row
        return last

    def previous_run(self, before: Optional[dt.date] = None) -> Optional[Dict[str, Any]]:
        """The most recent recorded run strictly before ``before``.

        The engine holds no live position state -- every run rebuilds its view
        from the store -- so this row is the entire memory the next run has.
        Both the open book and the previous screen come out of it, and they are
        read together because this is a full scan of a file that grows without
        bound; doing it twice per run was pure waste.

        Rows whose date will not parse are skipped rather than crashing the
        run: one bad line must not become "the engine holds nothing".
        """
        latest: Optional[Dict[str, Any]] = None
        latest_date: Optional[dt.date] = None
        for row in self.iter_rows():
            raw = row.get("date")
            try:
                when = dt.date.fromisoformat(str(raw)[:10])
            except (TypeError, ValueError):
                continue
            if before is not None and when >= before:
                continue
            if latest_date is None or when >= latest_date:
                latest_date, latest = when, row
        return latest

    def open_book(self, before: Optional[dt.date] = None) -> List[str]:
        """Names the most recent recorded run issued as BUY.

        Stage 6's exit band needs it: a name is kept while it stays inside the
        wider band, which cannot be evaluated without knowing whether it was
        held.

        Returns the empty list when nothing has been recorded yet, which is the
        correct starting state rather than an error -- a first run holds
        nothing.
        """
        row = self.previous_run(before=before)
        if not row:
            return []
        book = list(row.get("signals_generated") or [])
        # Positions the previous run could not evaluate but did not close. A
        # suspended name has no price to exit at and a name dropped from the
        # universe is still tradeable, so both are still held -- and both used
        # to leave the book simply by not being in `signals_generated`.
        seen = set(book)
        for directive in row.get("position_directives") or []:
            if not isinstance(directive, dict):
                continue
            ticker = str(directive.get("ticker") or "")
            if not ticker or ticker in seen:
                continue
            if str(directive.get("action") or "").startswith("hold"):
                book.append(ticker)
                seen.add(ticker)
        return book

    def shown_slate(self, before: Optional[dt.date] = None) -> List[Dict[str, Any]]:
        """The screen the most recent recorded run produced, in order.

        Empty for a run recorded before the slate was part of the record, which
        is the correct answer: there is no previous screen to carry, so the next
        one is chosen fresh. It is not an error and must not be inferred from
        `signals_generated` -- that is the book, which is a different list.
        """
        row = self.previous_run(before=before)
        return list(row.get("slate_shown") or []) if row else []

    def signals_for(self, ticker: str) -> List[Dict[str, Any]]:
        """Every run that produced a signal for one ticker -- the audit question."""
        return [r for r in self.iter_rows() if ticker in (r.get("signals_generated") or [])]


def row_from_output(
    output: FinalSignalOutput,
    context: RunContext,
    funnel: Dict[str, int],
    duration_ms: float,
    error: Optional[str] = None,
    train_sessions: Optional[int] = None,
) -> LedgerRow:
    """Flatten a completed run into its permanent record.

    Captures enough to answer "why did the system say this?" six months later:
    the config hash, the engine version, the regime, the full funnel, and every
    scored name with its factor values -- not just the winners.
    """
    scored: List[Dict[str, Any]] = []
    for rec in list(output.recommendations) + list(output.watchlist):
        scored.append(
            {
                "ticker": rec.ticker,
                "decision": rec.decision.value,
                "composite_score": rec.composite_score,
                "percentile": rec.universe_percentile,
                # BOTH, because they are different numbers and only one of them
                # decides anything. `rank` is the display position among the
                # defended survivors; `model_rank` is the name's place in the
                # full eligible universe and is the only input to Stage 6's
                # admission band. The record carried `rank` alone, so it could
                # not answer "was this name inside the band?" -- the single
                # question the whole hysteresis design turns on -- and every
                # reconstruction from the ledger silently used the wrong one.
                "rank": rec.rank,
                "model_rank": rec.model_rank,
                "sector": rec.sector,
                "last_close": rec.last_close,
                "entry_zone": list(rec.entry_zone) if rec.entry_zone else None,
                "stop": rec.initial_stop,
                "target_1": rec.target_1,
                "target_2": rec.target_2,
                "strength_band": rec.signal_strength_band.value,
            }
        )

    regime = output.regime_state
    return LedgerRow(
        trial_id=context.trial_id,
        run_id=output.run_id,
        date=output.as_of_date,
        logged_at=dt.datetime.now(),
        engine_version=output.engine_version,
        schema_version=context.schema_version,
        config_version=output.config_version,
        model_fingerprint=model_fingerprint(train_sessions),
        mode=context.mode,
        regime_state={
            "bucket": regime.regime_bucket,
            "trend": regime.trend_regime.value,
            "vol_tercile": regime.vol_tercile.value,
            "vol_context": regime.vol_context.value,
            "breadth_pct": regime.breadth_pct_above_ma,
            "transition": regime.transition_flag,
            "momentum_multiplier": regime.momentum_multiplier,
            "allow_new_entries": regime.allow_new_entries,
        },
        eligible_universe_size=funnel.get("passed_eligibility", 0),
        universe_considered=funnel.get("universe_considered", 0),
        stocks_scored=scored,
        signals_generated=[r.ticker for r in output.recommendations],
        watchlist_generated=[r.ticker for r in output.watchlist],
        slate_shown=[e.model_dump(mode="json") for e in output.slate],
        position_directives=list(output.position_directives),
        no_trade=output.no_trade is not None,
        new_entries_blocked=output.new_entries_blocked,
        no_trade_reason=output.no_trade.reason if output.no_trade else None,
        gate_counts=dict(funnel),
        data_quality_flags=list(output.data_quality_flags),
        survivorship_risk=bool(output.manifest.survivorship_risk) if output.manifest else False,
        stage_timings_ms=dict(output.stage_timings_ms),
        duration_ms=round(duration_ms, 1),
        error=error,
    )
