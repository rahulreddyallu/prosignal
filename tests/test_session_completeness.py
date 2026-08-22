"""A stored session must be complete before the fetcher calls it done.

`known_sessions()` returns the dates NSE published an INDEX file for, and it
gated the price backfill. The two are not the same thing. A day whose index
file was written and whose bhavcopy was not got marked fetched and was never
retried, because nothing ever re-examined a date already in the store.

The store carried exactly that: 2026-02-05, a Thursday with 145 index rows --
the same count as the sessions either side of it -- and zero equity prices.
Every 63-session window from 2026-02-05 through 2026-05-12 was computed across
the hole. The `keep="last"` supersession in DataStore.write is correct, but it
never fires for prices, because the fetch that would carry the correction is
skipped before a row is ever requested.
"""

from __future__ import annotations

import ast
import builtins
import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "prosignal"


# ---------------------------------------------------------------- calendar
def _store(tmp_path, index_dates, price_dates):
    from prosignal.data.store import DataStore

    store = DataStore(tmp_path / "curated", tmp_path / "snapshots")
    store.write_indices(pd.DataFrame({
        "date": pd.to_datetime(list(index_dates)),
        "index_name": ["NIFTY 50"] * len(index_dates),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0,
    }))
    if price_dates:
        store.write_prices(pd.DataFrame({
            "date": pd.to_datetime(list(price_dates)),
            "symbol": ["AAA"] * len(price_dates),
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
            "volume": 100, "turnover": 100.0, "series": "EQ",
            "source": "nse_archives",
        }))
    return store


def test_a_weekend_index_file_with_no_prices_is_not_a_session(tmp_path):
    """NSE's archive serves these for days the market never opened. Counted as
    sessions they shorten every window measured in sessions."""
    sunday = dt.date(2023, 6, 4)
    monday = dt.date(2023, 6, 5)
    store = _store(tmp_path, [sunday, monday], [monday])
    assert sunday.weekday() == 6
    assert store.known_sessions() == [monday]


def test_a_weekend_session_that_actually_traded_is_kept(tmp_path):
    """NSE has held real Saturday sessions. Prices are what distinguishes one
    from an archive artefact."""
    saturday = dt.date(2023, 11, 4)
    store = _store(tmp_path, [saturday], [saturday])
    assert saturday.weekday() == 5
    assert store.known_sessions() == [saturday]


def test_a_weekday_with_no_prices_stays_in_the_calendar_as_a_gap(tmp_path):
    """It must remain visible. Dropping it would make the store look complete
    and cost the refetch."""
    thursday = dt.date(2026, 2, 5)
    store = _store(tmp_path, [thursday], [])
    assert thursday.weekday() == 3
    assert store.known_sessions() == [thursday]
    assert store.price_sessions() == []


# ------------------------------------------------------------- fetch gate
def _ingestor(tmp_path):
    from prosignal.config.loader import load_config
    from prosignal.data.ingest import DataIngestor

    config = load_config()
    config.paths.curated = tmp_path / "curated"
    config.paths.snapshots = tmp_path / "snapshots"
    return DataIngestor(config)


def test_a_session_missing_its_prices_is_rescheduled(tmp_path):
    from prosignal.data.ingest import IngestOptions

    days = pd.bdate_range("2026-01-05", "2026-02-20").date.tolist()
    hole = dt.date(2026, 2, 5)
    assert hole in days
    _store(tmp_path, days, [d for d in days if d != hole])

    todo = _ingestor(tmp_path)._sessions_to_fetch(
        dt.date(2026, 2, 20), 30, IngestOptions()
    )
    assert hole in todo, (
        "a session with an index file but no prices was treated as fetched"
    )


def test_a_complete_session_is_not_refetched(tmp_path):
    from prosignal.data.ingest import IngestOptions

    days = pd.bdate_range("2026-01-05", "2026-02-20").date.tolist()
    _store(tmp_path, days, days)
    todo = _ingestor(tmp_path)._sessions_to_fetch(
        dt.date(2026, 2, 20), 30, IngestOptions()
    )
    assert not [d for d in todo if d in set(days)], (
        "complete sessions were rescheduled; every ingest would refetch history"
    )


def test_delivery_is_not_part_of_the_completeness_test(tmp_path):
    """NSE stopped serving delivery for pre-2021 dates. Requiring it would
    re-probe eight years of sessions on every ingest and never succeed."""
    src = (SRC / "data" / "ingest.py").read_text(encoding="utf-8")
    gate = src.split("have_index: set")[1].split("\n\n")[0]
    assert "delivery_sessions" not in gate and "read_delivery" not in gate


# ----------------------------------------------------- the NameError class
def _unbound_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))

    # Everything the function bodies sit inside, so module scope can be read
    # off as "bound somewhere, but not inside a function". Collecting only
    # tree.body would miss the common `try: import rich / except ImportError`
    # and anything bound under a module-level `if`.
    in_function = {
        n
        for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        for n in ast.walk(fn)
        if n is not fn
    }

    module = {"self", "cls", "__file__", "__name__", "__doc__"} | set(dir(builtins))
    for n in ast.walk(tree):
        if n in in_function:
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            module.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for alias in n.names:
                module.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            module.add(n.name)

    # A closure legitimately reads its enclosing function's locals, and the
    # enclosing walk already covers the nested body with those names bound --
    # so analyse only functions that are not themselves nested in one.
    nested = {
        inner
        for outer in ast.walk(tree)
        if isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef))
        for inner in ast.walk(outer)
        if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef))
        and inner is not outer
    }

    bad: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn in nested:
            continue
        a = fn.args
        bound = {x.arg for x in a.args + a.kwonlyargs + getattr(a, "posonlyargs", [])}
        if a.vararg:
            bound.add(a.vararg.arg)
        if a.kwarg:
            bound.add(a.kwarg.arg)
        for s in ast.walk(fn):
            if isinstance(s, ast.Name) and isinstance(s.ctx, (ast.Store, ast.Del)):
                bound.add(s.id)
            elif isinstance(s, (ast.Import, ast.ImportFrom)):
                for alias in s.names:
                    bound.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(s, ast.ExceptHandler) and s.name:
                bound.add(s.name)
            elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(s.name)
            elif isinstance(s, ast.arg):
                bound.add(s.arg)
        for s in ast.walk(fn):
            if (isinstance(s, ast.Name) and isinstance(s.ctx, ast.Load)
                    and s.id not in bound and s.id not in module):
                bad.append(f"{path.name}:{s.lineno} {fn.name}() reads unbound {s.id!r}")
    return bad


def test_no_function_reads_a_name_it_never_receives():
    """Extracting a body into a helper and forgetting to pass one of its
    parameters has now shipped twice from this package -- `_ingest_csv_feeds`
    losing `opts` and `symbols`, and `_run_locked` losing `requested_date`.
    Both crashed `data ingest` on every invocation and neither was caught by a
    test, because nothing imports a module and no unit test drives the CLI
    entry point end to end.
    """
    found: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        found.extend(_unbound_names(path))
    assert not found, "unbound names:\n" + "\n".join(found)
