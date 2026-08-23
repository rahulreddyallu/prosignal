"""What the engine said before, and what moved since.

Every completed run is already written to the ledger -- date, the names
admitted, the names monitored, the regime and the full funnel -- so the history
this builds is a read over a record that has existed all along rather than new
storage bolted on.

WHAT CANNOT BE DERIVED. The ledger keeps which names were admitted, not why.
Factor loadings are not retained per name, so a change log cannot honestly say
"momentum improved" -- the values that would prove it were never written down.
Status transitions and membership are recorded and are reported; the reasons
behind them are not, and are not guessed at.

POSITION IS ALSO NOT COMPARED. Runs logged before the interface rebuild ordered
their lists by the penalised score, and runs after it order by model rank.
Comparing a position across that boundary would manufacture movement out of a
change in how the list was sorted, so only membership and status are compared,
both of which mean the same thing on either side of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

BUY = "BUY"
WATCH = "WATCH"


@dataclass(frozen=True)
class Day:
    """One trading date's outcome, from the last run made for that date."""

    date: str
    run_id: Optional[str]
    buys: List[str] = field(default_factory=list)
    watch: List[str] = field(default_factory=list)
    regime: str = ""
    allows_new_positions: bool = True
    universe: Optional[int] = None
    logged_at: Optional[str] = None
    #: Per-name detail as it stood that day: the close the signal was formed
    #: on, the stop and the targets. Keyed by ticker. The ledger has carried
    #: this since October 2023, which is why past days can be followed up
    #: without having recorded anything new for the purpose.
    detail: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def buy_count(self) -> int:
        return len(self.buys)

    @property
    def watch_count(self) -> int:
        return len(self.watch)


def _row(record: Any) -> Dict[str, Any]:
    return record if isinstance(record, dict) else dict(getattr(record, "__dict__", {}))


def _regime_label(state: Dict[str, Any]) -> str:
    trend = str(state.get("trend") or "").strip()
    vol = str(state.get("vol_tercile") or "").strip()
    if trend and vol:
        return f"{trend}, {vol.lower()} volatility"
    return trend or "Unknown"


def load_days(records: Iterable[Any], *, limit: int = 30,
              since: Optional[str] = None) -> List[Day]:
    """Collapse the ledger to one entry per date, newest first.

    A date is run many times -- the store had 98 rows for its newest date --
    and only the last one for a date reflects what the engine finally said.
    Ordering by the logged timestamp rather than by file position keeps that
    true even when rows are appended out of order.
    """
    best: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for record in records:
        row = _row(record)
        date = row.get("date")
        if not date or row.get("error"):
            continue
        stamp = str(row.get("logged_at") or "")
        if since and stamp <= since:
            continue
        held = best.get(str(date))
        if held is None or stamp >= held[0]:
            best[str(date)] = (stamp, row)

    days: List[Day] = []
    for date in sorted(best, reverse=True)[:limit]:
        _, row = best[date]
        regime = row.get("regime_state") or {}
        gates = row.get("gate_counts") or {}
        detail = {
            str(entry.get("ticker")): entry
            for entry in (row.get("stocks_scored") or [])
            if isinstance(entry, dict) and entry.get("ticker")
        }
        days.append(Day(
            date=date,
            run_id=row.get("run_id"),
            detail=detail,
            buys=[str(t) for t in (row.get("signals_generated") or [])],
            watch=[str(t) for t in (row.get("watchlist_generated") or [])],
            regime=_regime_label(regime),
            allows_new_positions=bool(regime.get("allow_new_entries", True)),
            universe=gates.get("universe_considered"),
            logged_at=row.get("logged_at"),
        ))
    return days


def changes(today: Day, previous: Optional[Day], *, slots: int = 5) -> Dict[str, Any]:
    """What moved between two runs.

    Only the names either run put on the screen are compared. The engine
    monitors dozens, and a name drifting between rank 38 and rank 41 is not a
    change anyone needs told about -- it never appeared and still does not.
    """
    if previous is None:
        return {
            "available": False,
            "reason": "This is the earliest run on record, so there is nothing "
                      "to compare it against.",
            "entered": [], "left": [], "promoted": [], "demoted": [],
        }

    now = _slate(today, slots)
    was = _slate(previous, slots)

    entered = [t for t in now if t not in was]
    left = [t for t in was if t not in now]
    promoted = [t for t in now
                if now[t] == BUY and was.get(t) == WATCH]
    demoted = [t for t in now
               if now[t] == WATCH and was.get(t) == BUY]

    return {
        "available": True,
        "compared_with": previous.date,
        "entered": entered,
        "left": left,
        "promoted": promoted,
        "demoted": demoted,
        "unchanged": [t for t in now if t in was and now[t] == was[t]],
        "summary": _summarise(entered, left, promoted, demoted, previous.date),
        # Stated rather than implied. A change log that silently omitted the
        # reason would read as though there wasn't one.
        "reason_note": (
            "The ledger records which names were selected, not the factor "
            "values behind them, so the reason a name moved is not recoverable "
            "from past runs."
        ),
    }


def _slate(day: Day, slots: int) -> Dict[str, str]:
    """Reconstruct what that day's screen showed: buys first, then near misses."""
    out: Dict[str, str] = {}
    for ticker in day.buys[:slots]:
        out[ticker] = BUY
    for ticker in day.watch:
        if len(out) >= slots:
            break
        if ticker not in out:
            out[ticker] = WATCH
    return out


def slate_picks(day: Day, slots: int = 5) -> List[Dict[str, Any]]:
    """That day's screen, with the levels it was showing at the time."""
    picks: List[Dict[str, Any]] = []
    for position, (ticker, status) in enumerate(_slate(day, slots).items(), 1):
        entry = day.detail.get(ticker) or {}
        picks.append({
            "position": position,
            "ticker": ticker,
            "status": status,
            "signal_price": entry.get("last_close"),
            "stop": entry.get("stop"),
            "target_1": entry.get("target_1"),
            "target_2": entry.get("target_2"),
            "strength": entry.get("strength_band"),
            "sector": entry.get("sector") if entry.get("sector") != "Unknown" else None,
        })
    return picks


def _plural(items: Sequence[str], one: str, many: str) -> str:
    return one if len(items) == 1 else many


def _summarise(entered, left, promoted, demoted, prev_date: str) -> str:
    parts: List[str] = []
    if entered:
        parts.append(f"{len(entered)} {_plural(entered, 'name', 'names')} entered")
    if left:
        parts.append(f"{len(left)} dropped out")
    if promoted:
        parts.append(f"{len(promoted)} moved from watch to buy")
    if demoted:
        parts.append(f"{len(demoted)} moved from buy to watch")
    if not parts:
        return f"The shortlist is unchanged from {prev_date}."
    text = parts[0][0].upper() + parts[0][1:]
    if len(parts) > 1:
        text = ", ".join([text] + parts[1:-1]) + f" and {parts[-1]}"
    return f"{text} since {prev_date}."


def build_history(
    records: Iterable[Any],
    *,
    limit: int = 30,
    slots: int = 5,
    company_names: Optional[Dict[str, str]] = None,
    since: Optional[str] = None,
) -> Dict[str, Any]:
    """The History view's data."""
    names = company_names or {}
    days = load_days(records, limit=limit, since=since)
    if not days:
        return {
            "days": [], "latest_changes": None, "cleared_at": since,
            "note": (
                "History was cleared. Runs from here on will be recorded and "
                "will appear on this page."
                if since else "No completed runs have been recorded yet."
            ),
        }

    def label(ticker: str) -> Dict[str, str]:
        full = names.get(ticker) or ticker
        for suffix in (" Limited", " Ltd.", " Ltd", " LIMITED"):
            if full.endswith(suffix):
                full = full[: -len(suffix)].strip()
                break
        return {"ticker": ticker, "company": full}

    latest = changes(days[0], days[1] if len(days) > 1 else None, slots=slots)
    for key in ("entered", "left", "promoted", "demoted"):
        latest[key] = [label(t) for t in latest.get(key, [])]

    out_days: List[Dict[str, Any]] = []
    for index, day in enumerate(days):
        prior = days[index + 1] if index + 1 < len(days) else None
        delta = changes(day, prior, slots=slots)
        slate = _slate(day, slots)
        out_days.append({
            "date": day.date,
            "regime": day.regime,
            "allows_new_positions": day.allows_new_positions,
            "buy_count": day.buy_count,
            "watch_count": day.watch_count,
            "shown": [
                {**label(t), "status": status} for t, status in slate.items()
            ],
            "change_summary": delta.get("summary") if delta.get("available") else None,
        })

    return {
        "days": out_days,
        "latest_changes": latest,
        "cleared_at": since,
        "note": "",
    }


def runs_for_ticker(
    records: Iterable[Any],
    ticker: str,
    *,
    slots: int = 5,
    since: Optional[str] = None,
    limit: int = 400,
) -> List[Dict[str, Any]]:
    """Every run that put this name in front of the reader, newest first.

    Membership of the displayed slate is the test, not membership of the
    watchlist. The engine monitors dozens of names it never surfaces, and a
    run that ranked something 41st did not tell anyone about it -- listing it
    as a past call would invent a history that never reached a screen.
    """
    wanted = str(ticker).upper()
    out: List[Dict[str, Any]] = []
    for day in load_days(records, limit=limit, since=since):
        slate = _slate(day, slots)
        status = slate.get(wanted)
        if status is None:
            continue
        entry = day.detail.get(wanted) or {}
        out.append({
            "date": day.date,
            "status": status,
            "regime": day.regime,
            "ticker": wanted,
            "signal_price": entry.get("last_close"),
            "stop": entry.get("stop"),
            "target_1": entry.get("target_1"),
            "strength": entry.get("strength_band"),
        })
    return out
