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
from .core.errors import AmbiguousLedgerHistory, LedgerError
from .core.logging import get_logger

__all__ = ["Ledger", "row_from_output", "AmbiguousLedgerHistory"]

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

    @staticmethod
    def _book_identity(row: Dict[str, Any]) -> tuple:
        """What a row claims was held and shown. Two rows agreeing on this
        describe the same state, whatever else differs between them."""
        book = tuple(sorted(str(s) for s in (row.get("signals_generated") or [])))
        slate = tuple(
            str(e.get("ticker"))
            for e in (row.get("slate_shown") or [])
            if isinstance(e, dict) and e.get("ticker")
        )
        return book, slate

    def _rows_on(self, when: dt.date, mode: Optional[str]) -> List[Dict[str, Any]]:
        """Every row for one date and lineage, in a deterministic order.

        Ordered by `(logged_at, run_id)` rather than by position in the file.
        File order is an artefact of when a process happened to flush; two
        clones of the same ledger must resolve the same book.
        """
        out = []
        for row in self.iter_rows():
            if self._row_date(row) != when:
                continue
            if mode is not None and str(row.get("mode") or "live") != mode:
                continue
            out.append(row)
        return sorted(out, key=lambda r: (str(r.get("logged_at") or ""),
                                          str(r.get("run_id") or "")))

    @staticmethod
    def _row_date(row: Dict[str, Any]) -> Optional[dt.date]:
        try:
            return dt.date.fromisoformat(str(row.get("date"))[:10])
        except (TypeError, ValueError):
            return None

    def previous_run(
        self,
        before: Optional[dt.date] = None,
        mode: Optional[str] = "live",
    ) -> Optional[Dict[str, Any]]:
        """The most recent recorded run strictly before ``before``.

        The engine holds no live position state -- every run rebuilds its view
        from the store -- so this row is the entire memory the next run has.
        Both the open book and the previous screen come out of it, and they are
        read together because this is a full scan of a file that grows without
        bound; doing it twice per run was pure waste.

        TWO THINGS THIS GUARDS, both of which it used to get wrong.

        LINEAGE. Rows are filtered to one ``mode`` -- the run's own. A live run
        reads live history; a replay of a past date reads replay history. They
        used to share one stream, so a backfill appended after a live run became
        that run's successor and the next live session inherited a book from a
        reconstruction. `mode` has existed on the row since v1 and was written
        as the literal "live" on every path, so nothing could tell them apart.
        Pass ``mode=None`` to read across every lineage, which is what a
        reporting caller wants and a pipeline caller never does.

        AMBIGUITY. The selection rule was `if latest_date is None or when >=
        latest_date`, so within a date the last line in the FILE won. On the
        shipped ledger 80 dates carried more than one run and 2026-08-18 carried
        676 of them recording seven different books. There is no fact of the
        matter about what was held on such a date, so this raises
        :class:`AmbiguousLedgerHistory` rather than choosing. Rows that agree
        about the book -- the ordinary case of an operator pressing SCAN twice
        -- are not a conflict, and the newest is returned.

        Rows whose date will not parse are skipped rather than crashing the
        run: one bad line must not become "the engine holds nothing".
        """
        latest_date: Optional[dt.date] = None
        for row in self.iter_rows():
            when = self._row_date(row)
            if when is None:
                continue
            if before is not None and when >= before:
                continue
            if mode is not None and str(row.get("mode") or "live") != mode:
                continue
            if latest_date is None or when > latest_date:
                latest_date = when
        if latest_date is None:
            return None

        rows = self._rows_on(latest_date, mode)
        if not rows:
            # Unreachable while `latest_date` is chosen under the same `mode`
            # filter `_rows_on` applies -- which is the point. Mutating either
            # filter away makes the two disagree, and the symptom would be an
            # IndexError on `rows[-1]` rather than anything a reader could act
            # on. Found by mutation-testing the lineage filter.
            raise LedgerError(
                f"no {mode or 'any'}-mode run recorded for "
                f"{latest_date.isoformat()}, yet that date was selected as the "
                f"most recent one. The date scan and the row fetch are "
                f"filtering differently.",
                date=latest_date.isoformat(), mode=str(mode))
        identities = {self._book_identity(r) for r in rows}
        if len(identities) > 1:
            books = sorted({i[0] for i in identities})
            raise AmbiguousLedgerHistory(
                f"{len(rows)} runs are recorded for {latest_date.isoformat()} "
                f"and they disagree about the book: "
                + "; ".join(
                    "[" + (", ".join(b) if b else "empty") + "]" for b in books[:6]
                )
                + (f" (+{len(books) - 6} more)" if len(books) > 6 else "")
                + ". The open book is the engine's entire position memory, so "
                  "there is nothing to fall back to -- resolve the ledger for "
                  "that date (keep one run, or re-record it) before running "
                  "against it.",
                date=latest_date.isoformat(),
                runs=len(rows),
                distinct_books=len(books),
                run_ids=[str(r.get("run_id")) for r in rows[:8]],
            )
        return rows[-1]

    #: A run recorded this many calendar days after the session it scores is
    #: still the live run for it. One, not zero: the cron fires in the evening
    #: and a run that starts at 23:50 finishes tomorrow.
    LIVE_RECORDING_LAG_DAYS = 1

    @classmethod
    def _observed_mode(cls, row: Dict[str, Any]) -> Optional[str]:
        """What the row's own timestamps say it was, or None if they cannot say.

        The ONLY evidence in the record that separates a session the market
        produced from a re-derivation of one. `mode` was written as the literal
        "live" on every path, so it carries no information; `logged_at` against
        `date` carries all of it. A run recorded weeks after the session it
        scores is a backfill whatever its `mode` field claims.
        """
        market = cls._row_date(row)
        try:
            logged = dt.date.fromisoformat(str(row.get("logged_at"))[:10])
        except (TypeError, ValueError):
            return None
        if market is None:
            return None
        lag = (logged - market).days
        if lag < 0:
            return None              # recorded before the session: unreadable
        return "live" if lag <= cls.LIVE_RECORDING_LAG_DAYS else "replay"

    def lineage_audit(self) -> Dict[str, Any]:
        """What the ledger actually contains, before anything is changed.

        Read-only. Answers the question `mode` was supposed to answer and
        could not: how much of this record is the engine running, and how much
        is the engine being re-run.
        """
        live_dates: Dict[dt.date, List[Dict[str, Any]]] = {}
        replay_dates: set = set()
        unknown = 0
        for row in self.iter_rows():
            observed = self._observed_mode(row)
            when = self._row_date(row)
            if observed is None or when is None:
                unknown += 1
                continue
            if observed == "live":
                live_dates.setdefault(when, []).append(row)
            else:
                replay_dates.add(when)
        conflicted = []
        for when, rows in sorted(live_dates.items()):
            identities = {self._book_identity(r) for r in rows}
            if len(identities) > 1:
                conflicted.append({
                    "date": when.isoformat(),
                    "runs": len(rows),
                    "distinct_books": len({i[0] for i in identities}),
                    "config_versions": sorted({str(r.get("config_version"))
                                               for r in rows}),
                })
        return {
            "rows": sum(1 for _ in self.iter_rows()),
            "market_dates": len(set(live_dates) | replay_dates),
            "recorded_on_the_day": sorted(d.isoformat() for d in live_dates),
            "backfilled_only": len(replay_dates - set(live_dates)),
            "undatable_rows": unknown,
            "live_dates_that_conflict": conflicted,
        }

    def repair_lineage(self, *, dry_run: bool = True) -> Dict[str, Any]:
        """Stamp each row with the lineage its own timestamps prove, once.

        NOTHING IS DELETED. Every row keeps its measurements, its trial id and
        its place in the file; what changes is one label that was never
        populated with anything but the constant "live". Deleting the
        contaminated rows was the obvious move and it is the wrong one: the
        honest trial count feeds the Deflated Sharpe directly, `Ledger.append`
        is fatal-on-failure precisely so that count cannot be corrupted, and
        removing 1,365 rows to tidy a lineage field would corrupt it far more
        thoroughly than a wrong label ever did.

        THREE OUTCOMES, and the third is the one that matters.

        `live`    -- recorded within LIVE_RECORDING_LAG_DAYS of the session it
                     scores. This is the engine running.
        `replay`  -- recorded later. A re-derivation of a past session, which
                     is a legitimate and useful thing to do and is not a
                     forward observation of anything.
        `quarantine` -- recorded on the day, and DISAGREEING with another run
                     recorded on the same day about what was held. These cannot
                     be resolved and this method does not try: the same date
                     under the SAME config_version produced up to five different
                     books, so the code moved underneath, and
                     `model_fingerprint` -- the field that exists to catch
                     exactly that -- is null on 1,733 of 1,942 rows. There is no
                     recorded property that reconstructs which book was real.
                     Guessing would put a fabricated position history under
                     every hysteresis decision that reads it, so they are set
                     aside instead: still counted, still readable, out of the
                     lineage.

        Returns the summary either way; `dry_run=False` writes. A backup of
        every file is written beside it first.
        """
        import shutil

        summary = {"live": 0, "replay": 0, "quarantine": 0, "unchanged": 0,
                   "undatable": 0, "files": [], "quarantined_dates": [],
                   "dry_run": dry_run}

        # Which same-day dates disagree with themselves. Decided BEFORE any row
        # is rewritten, so the classification is a function of the file as it
        # stands rather than of the order rows happen to be visited in.
        by_day: Dict[dt.date, List[Dict[str, Any]]] = {}
        for row in self.iter_rows():
            if self._observed_mode(row) == "live":
                when = self._row_date(row)
                if when is not None:
                    by_day.setdefault(when, []).append(row)
        quarantined = {
            when for when, rows in by_day.items()
            if len({self._book_identity(r) for r in rows}) > 1
        }
        summary["quarantined_dates"] = sorted(d.isoformat() for d in quarantined)

        for path in sorted(self.dir.glob("runs-*.jsonl")):
            out_lines: List[str] = []
            changed = 0
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    out_lines.append(line)          # keep what cannot be parsed
                    continue
                observed = self._observed_mode(row)
                if observed is None:
                    summary["undatable"] += 1
                    out_lines.append(line)
                    continue
                when = self._row_date(row)
                target = ("quarantine"
                          if observed == "live" and when in quarantined
                          else observed)
                summary[target] += 1
                if str(row.get("mode") or "live") == target:
                    summary["unchanged"] += 1
                    out_lines.append(line)
                    continue
                row["mode"] = target
                # Why, on the row, so the change explains itself to the next
                # reader without reference to this docstring.
                row["mode_source"] = (
                    "repair_lineage: logged_at vs date"
                    + ("; same-day runs disagree about the book"
                       if target == "quarantine" else "")
                )
                changed += 1
                out_lines.append(json.dumps(row, separators=(",", ":"),
                                            sort_keys=True))
            summary["files"].append({"file": path.name, "rewritten": changed})
            if not dry_run and changed:
                backup = path.with_suffix(".jsonl.pre-lineage-repair")
                if not backup.exists():                  # never clobber a backup
                    shutil.copy2(path, backup)
                tmp = path.with_suffix(".jsonl.tmp")
                tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
                os.replace(str(tmp), str(path))
        return summary

    def conflicting_dates(self, mode: Optional[str] = "live") -> List[Dict[str, Any]]:
        """Every date whose recorded runs disagree about the book.

        The diagnostic behind :class:`AmbiguousLedgerHistory`: it answers "which
        days do I have to clean up" without running an analysis into the wall.
        """
        by_date: Dict[dt.date, List[Dict[str, Any]]] = {}
        for row in self.iter_rows():
            when = self._row_date(row)
            if when is None:
                continue
            if mode is not None and str(row.get("mode") or "live") != mode:
                continue
            by_date.setdefault(when, []).append(row)
        out: List[Dict[str, Any]] = []
        for when in sorted(by_date):
            rows = by_date[when]
            identities = {self._book_identity(r) for r in rows}
            if len(identities) > 1:
                out.append({
                    "date": when.isoformat(),
                    "runs": len(rows),
                    "distinct_books": len({i[0] for i in identities}),
                    "config_versions": sorted(
                        {str(r.get("config_version")) for r in rows}),
                })
        return out

    def open_book(self, before: Optional[dt.date] = None,
                  mode: Optional[str] = "live") -> List[str]:
        """Names the most recent recorded run issued as BUY.

        Stage 6's exit band needs it: a name is kept while it stays inside the
        wider band, which cannot be evaluated without knowing whether it was
        held.

        Returns the empty list when nothing has been recorded yet, which is the
        correct starting state rather than an error -- a first run holds
        nothing.
        """
        row = self.previous_run(before=before, mode=mode)
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

    def shown_slate(self, before: Optional[dt.date] = None,
                    mode: Optional[str] = "live") -> List[Dict[str, Any]]:
        """The screen the most recent recorded run produced, in order.

        Empty for a run recorded before the slate was part of the record, which
        is the correct answer: there is no previous screen to carry, so the next
        one is chosen fresh. It is not an error and must not be inferred from
        `signals_generated` -- that is the book, which is a different list.
        """
        row = self.previous_run(before=before, mode=mode)
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
                # THE PLAN, recorded with the decision rather than reconstructed
                # from the config later. A config is a moving object -- the
                # cadence, the planned hold and the expectancy study all change
                # -- so reading today's config to explain a trade issued in
                # March would describe a strategy that trade was never part of.
                # This is what makes the paper-trading record self-contained:
                # every row carries the frequency it was issued at, the hold it
                # was planned for and the expectation it was issued under.
                "trade_plan": (rec.trade_plan.model_dump(mode="json")
                               if rec.trade_plan else None),
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
