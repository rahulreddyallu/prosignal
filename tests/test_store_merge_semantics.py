"""A partial feed must degrade the store, never delete it.

`write_statements` called `replace_table`, so every write discarded whatever the
previous one had fetched and kept only the current batch. `fetch_statements` is
per-symbol and skips whatever the provider refuses -- which is correct, the
universe is wider than any statement feed -- so the table converged on whatever
the LAST fetch happened to return rather than on the union of everything ever
fetched.

That is why the store covered 200 symbols against a 750-name universe while the
feed itself serves the universe: probed live, Yahoo returned statements for 12
of 12 sampled symbols the store had nothing for, current to 2026-06-30. Nothing
was wrong with the feed. The store was deleting it.

Found by running a widening job against the real store and watching coverage
fall from 192 symbols to 25.
"""

from __future__ import annotations

import pandas as pd
import pytest

from prosignal.data.store import DataStore


def _stmt(symbols, period="2026-03-31", kind="quarterly", value=1.0):
    return pd.DataFrame({
        "symbol": list(symbols),
        "period_end": [pd.Timestamp(period)] * len(symbols),
        "kind": [kind] * len(symbols),
        "Net Income": [value] * len(symbols),
    })


@pytest.fixture
def store(tmp_path):
    return DataStore(tmp_path / "curated", tmp_path / "snapshots")


def test_a_second_batch_does_not_delete_the_first(store):
    """THE DEFECT. Two batches of a per-symbol feed must union, not replace."""
    store.write_statements(_stmt(["AAA", "BBB"]))
    store.write_statements(_stmt(["CCC"]))
    assert sorted(store.read_statements()["symbol"]) == ["AAA", "BBB", "CCC"]


def test_a_provider_that_serves_nothing_leaves_the_store_intact(store):
    """The failure mode that made this invisible: a feed returning nothing is
    the ordinary case for a symbol Yahoo does not cover, and it must not empty
    the table."""
    store.write_statements(_stmt(["AAA", "BBB"]))
    store.write_statements(pd.DataFrame(columns=["symbol", "period_end", "kind"]))
    assert sorted(store.read_statements()["symbol"]) == ["AAA", "BBB"]


def test_re_running_the_same_batch_is_idempotent(store):
    store.write_statements(_stmt(["AAA", "BBB"]))
    store.write_statements(_stmt(["AAA", "BBB"]))
    assert len(store.read_statements()) == 2


def test_a_restatement_supersedes_rather_than_duplicating(store):
    """Merged on (symbol, period_end, kind) with the incoming row winning."""
    store.write_statements(_stmt(["AAA"], value=1.0))
    store.write_statements(_stmt(["AAA"], value=9.0))
    out = store.read_statements()
    assert len(out) == 1
    assert float(out.iloc[0]["Net Income"]) == 9.0


def test_periods_and_kinds_are_separate_rows(store):
    """The key has to carry all three, or a quarterly result would overwrite an
    annual one for the same name."""
    store.write_statements(_stmt(["AAA"], period="2026-03-31", kind="quarterly"))
    store.write_statements(_stmt(["AAA"], period="2026-03-31", kind="annual"))
    store.write_statements(_stmt(["AAA"], period="2025-03-31", kind="quarterly"))
    assert len(store.read_statements()) == 3


def test_the_whole_file_feeds_still_replace(store):
    """`equity_master`, `sector_map` and `corporate_actions` are each fetched as
    one complete file, so replacing IS right for them -- a name NSE drops from
    its master should leave the store. This pins that the fix did not turn every
    feed into an append-only one."""
    store.write_equity_master(pd.DataFrame({
        "symbol": ["AAA", "BBB"], "company_name": ["A", "B"],
        "series": ["EQ", "EQ"], "listing_date": [pd.Timestamp("2020-01-01")] * 2,
        "paid_up_value": [1.0, 1.0], "face_value": [1.0, 1.0],
        "isin": ["I1", "I2"]}))
    store.write_equity_master(pd.DataFrame({
        "symbol": ["AAA"], "company_name": ["A"], "series": ["EQ"],
        "listing_date": [pd.Timestamp("2020-01-01")], "paid_up_value": [1.0],
        "face_value": [1.0], "isin": ["I1"]}))
    assert sorted(store.read_equity_master()["symbol"]) == ["AAA"], (
        "a delisted name must leave a whole-file feed"
    )


def test_the_three_annual_statements_fold_into_one_row_without_losing_fields():
    """`fetch_statements` emits THREE annual rows per symbol-year -- income
    statement, balance sheet and cash flow -- each carrying only its own fields
    and NaN elsewhere.

    A plain dedup on (symbol, period_end, kind) keeps whichever arrived last and
    silently discards the other two, so revenue and equity vanish and only the
    cash-flow columns survive. That is what the first version of this fix did:
    annual rows fell from 2,406 to 1,259 on the real store while symbol coverage
    rose, which is how it was caught.
    """
    import numpy as np
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())
    store = DataStore(tmp / "curated", tmp / "snapshots")
    three = pd.DataFrame({
        "symbol": ["AAA"] * 3,
        "period_end": [pd.Timestamp("2026-03-31")] * 3,
        "kind": ["annual"] * 3,
        "Total Revenue": [100.0, np.nan, np.nan],
        "Common Stock Equity": [np.nan, 50.0, np.nan],
        "Operating Cash Flow": [np.nan, np.nan, 20.0],
    })
    store.write_statements(three)
    out = store.read_statements()
    assert len(out) == 1
    row = out.iloc[0]
    assert row["Total Revenue"] == 100.0
    assert row["Common Stock Equity"] == 50.0
    assert row["Operating Cash Flow"] == 20.0, (
        "the cash-flow row must not be the only survivor"
    )


def test_the_fold_is_lossless_on_the_real_store(live_cfg):
    """Measured on the shipped table: 3,420 rows fold to 1,842 and the count of
    populated cells is unchanged at 28,904. Rows fall because NaN padding goes,
    not because data does."""
    store = DataStore(live_cfg.paths.curated, live_cfg.paths.snapshots)
    cur = store.read_statements()
    if cur is None or cur.empty:
        pytest.skip("no statements in this checkout")
    keys = ["symbol", "period_end", "kind"]
    folded = cur.groupby(keys, dropna=False, as_index=False).last()
    fields = [c for c in cur.columns if c not in keys]
    before = sum(int(cur[c].notna().sum()) for c in fields)
    after = sum(int(folded[c].notna().sum()) for c in fields)
    assert after == before, (
        f"folding lost {before - after} populated values"
    )
