"""Point-in-time fundamentals.

TWO SOURCES, ONE RULE: nothing is visible before it was disclosed.

  fundamentals.parquet carries a REAL `filing_date`. It is used as given.
  statements.parquet carries only `period_end`. A factor keyed on period end
  uses the number weeks or months before anybody could read it, which is
  lookahead that no backtest can see.

THE ASSUMED LAG IS MEASURED, NOT ASSERTED. The 3,504 rows that do carry a real
filing date give the disclosure-lag distribution directly:

    quarter end   n     p95    p99    max
    March        871     91    112    391      <- audited, slowest
    June         832     64    104    132
    September    880     45     57    170
    December     921     45     54     77

The statutory SEBI LODR deadline is 45 days (60 for the audited annual), and
only 84.1% of real filings were out by day 45 -- so a 45-day assumption would
have leaked on one row in six. The lag used here is the measured p99 per
quarter-end month, floored at 60 days. It is deliberately on the late side: a
lag that is too long makes a factor staler and weaker, a lag that is too short
manufactures alpha.
"""
from __future__ import annotations
import numpy as np, pandas as pd



#: Measured p99 of the real filing lag, by quarter-end month, floored at 60d.
DISCLOSURE_LAG_DAYS = {3: 112, 6: 104, 9: 60, 12: 60}
DEFAULT_LAG_DAYS = 112

#: A fundamental older than this is not a description of the company any more.
#: One annual cycle plus the audited lag plus a quarter of slack.
MAX_AGE_DAYS = 420


def _lag(period_end: pd.Series) -> pd.Series:
    m = pd.to_datetime(period_end).dt.month
    return m.map(DISCLOSURE_LAG_DAYS).fillna(DEFAULT_LAG_DAYS).astype("int64")


def _ttm(g: pd.DataFrame, cols, n=4) -> pd.DataFrame:
    """Trailing-twelve-month sums over the last `n` quarterly rows."""
    out = g[cols].rolling(n, min_periods=n).sum()
    return out


def build_records(store=None, D=None) -> pd.DataFrame:
    """One row per (symbol, effective_date) holding the company's state as it
    was knowable on that date. Flow items are TTM; stock items are as-reported.
    """
    recs = []

    # ---- source A: real filing dates, quarterly income statement -----------
    f = (store.read_fundamentals() if store is not None
         else pd.read_parquet(f"{D}/fundamentals.parquet"))
    if f is None or f.empty:
        f = pd.DataFrame(columns=["symbol", "filing_date", "period_end",
                                  "consolidated", "revenue", "net_profit",
                                  "profit_before_tax", "finance_costs",
                                  "depreciation", "total_income", "expenses",
                                  "shares_outstanding"])
    f["filing_date"] = pd.to_datetime(f["filing_date"])
    f["period_end"] = pd.to_datetime(f["period_end"])
    f = f.sort_values(["symbol", "period_end"])
    # One row per (symbol, period_end): consolidated preferred, and the EARLIEST
    # filing wins -- a restatement filed later must not backdate itself onto the
    # date the original was published.
    f["_pref"] = (f["consolidated"].astype(str).str.lower() == "consolidated").astype(int)
    f = (f.sort_values(["symbol", "period_end", "_pref", "filing_date"],
                       ascending=[True, True, False, True])
           .drop_duplicates(["symbol", "period_end"], keep="first"))
    flow = ["revenue", "net_profit", "profit_before_tax", "finance_costs",
            "depreciation", "total_income", "expenses"]
    for c in flow:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    g = f.groupby("symbol", sort=False)
    for c in flow:
        f["ttm_" + c] = g[c].transform(lambda s: s.rolling(4, min_periods=4).sum())
    a = pd.DataFrame({
        "symbol": f["symbol"], "effective_date": f["filing_date"],
        "period_end": f["period_end"], "src": "filed",
        "ttm_revenue": f["ttm_revenue"], "ttm_net_profit": f["ttm_net_profit"],
        "ttm_pbt": f["ttm_profit_before_tax"],
        "ttm_interest": f["ttm_finance_costs"],
        "ttm_depreciation": f["ttm_depreciation"],
        "shares": pd.to_numeric(f["shares_outstanding"], errors="coerce"),
    })
    recs.append(a)

    # ---- source B: period_end only, lagged by the measured p99 -------------
    s = (store.read_statements() if store is not None
         else pd.read_parquet(f"{D}/statements.parquet"))
    if s is None or s.empty:
        s = pd.DataFrame(columns=["symbol", "period_end", "kind"])
    s["period_end"] = pd.to_datetime(s["period_end"])
    s["effective_date"] = s["period_end"] + pd.to_timedelta(_lag(s["period_end"]), unit="D")
    s = s.sort_values(["symbol", "period_end"])
    num = lambda c: pd.to_numeric(s[c], errors="coerce") if c in s.columns else np.nan

    q = s[s["kind"] == "quarterly"].copy()
    if len(q):
        qg = q.groupby("symbol", sort=False)
        for src, dst in (("Total Revenue", "ttm_revenue"), ("Net Income", "ttm_net_profit"),
                         ("EBITDA", "ttm_ebitda"), ("EBIT", "ttm_ebit"),
                         ("Pretax Income", "ttm_pbt"),
                         ("Interest Expense", "ttm_interest"),
                         ("Gross Profit", "ttm_gross_profit")):
            if src in q.columns:
                q[dst] = qg[src].transform(lambda x: pd.to_numeric(x, errors="coerce")
                                           .rolling(4, min_periods=4).sum())
        keep = ["symbol", "effective_date", "period_end"] + [
            c for c in q.columns if c.startswith("ttm_")]
        b = q[keep].copy(); b["src"] = "lagged_q"
        recs.append(b)

    ann = s[s["kind"] == "annual"].copy()
    if len(ann):
        ren = {"Total Revenue": "ttm_revenue", "Net Income": "ttm_net_profit",
               "EBITDA": "ttm_ebitda", "EBIT": "ttm_ebit",
               "Pretax Income": "ttm_pbt", "Interest Expense": "ttm_interest",
               "Gross Profit": "ttm_gross_profit",
               "Operating Cash Flow": "ttm_ocf", "Free Cash Flow": "ttm_fcf",
               "Capital Expenditure": "ttm_capex",
               "Common Stock Equity": "equity", "Total Assets": "total_assets",
               "Total Debt": "total_debt", "Net Debt": "net_debt",
               "Invested Capital": "invested_capital",
               "Cash And Cash Equivalents": "cash",
               "Current Assets": "current_assets",
               "Current Liabilities": "current_liabilities",
               "Inventory": "inventory", "Accounts Receivable": "receivables",
               "Tangible Book Value": "tangible_book",
               "Ordinary Shares Number": "shares"}
        cols = {k: v for k, v in ren.items() if k in ann.columns}
        c = ann[["symbol", "effective_date", "period_end"] + list(cols)].rename(columns=cols)
        for col in cols.values():
            c[col] = pd.to_numeric(c[col], errors="coerce")
        c["src"] = "lagged_a"
        recs.append(c)

    out = pd.concat(recs, ignore_index=True)
    out = out.sort_values(["symbol", "effective_date", "period_end"])
    # Merge every record a symbol has published, forward-filling each FIELD
    # separately: an annual balance sheet and a later quarterly income statement
    # describe the same company at different resolutions, and a name should not
    # lose its book value because its newest disclosure was an income statement.
    fields = [c for c in out.columns
              if c not in ("symbol", "effective_date", "period_end", "src")]
    out = out.groupby(["symbol", "effective_date"], as_index=False).last()
    out = out.sort_values(["symbol", "effective_date"])
    for c in fields:
        out[c] = out.groupby("symbol", sort=False)[c].ffill()
    # Per-field recency, so a staleness gate can be applied to the FIELD a
    # factor actually uses rather than to the row it happens to sit on.
    out["_ed"] = out["effective_date"]
    for c in ("equity", "total_assets", "ttm_net_profit", "ttm_revenue"):
        if c in out.columns:
            seen = out["_ed"].where(out[c].notna())
            out["age_ref_" + c] = seen.groupby(out["symbol"]).ffill()
    return out.drop(columns=["_ed"]).reset_index(drop=True)


def asof_panel(records: pd.DataFrame, dates, symbols) -> dict:
    """As-of join: for each (date, symbol) the newest record already disclosed.

    Returns {field: (T x N) DataFrame} plus `fund_age_days`.
    """
    dates = pd.DatetimeIndex(dates)
    fields = [c for c in records.columns
              if c not in ("symbol", "effective_date", "period_end", "src")
              and not c.startswith("age_ref_")]
    out = {f: pd.DataFrame(np.nan, index=dates, columns=symbols, dtype="float32")
           for f in fields}
    age = pd.DataFrame(np.nan, index=dates, columns=symbols, dtype="float32")
    di = pd.Series(np.arange(len(dates)), index=dates)
    for sym, g in records.groupby("symbol", sort=False):
        if sym not in out[fields[0]].columns:
            continue
        g = g.sort_values("effective_date")
        # index of the first session on or after each effective date
        pos = dates.searchsorted(g["effective_date"].to_numpy(), side="left")
        for f in fields:
            v = g[f].to_numpy("float64")
            col = np.full(len(dates), np.nan)
            for p, x in zip(pos, v):
                if p < len(dates):
                    col[p] = x
            s = pd.Series(col).ffill().to_numpy()
            out[f][sym] = s.astype("float32")
        eff = np.full(len(dates), np.nan)
        for p, d in zip(pos, g["effective_date"].to_numpy("datetime64[ns]").astype("int64")):
            if p < len(dates):
                eff[p] = d
        eff = pd.Series(eff).ffill().to_numpy()
        age[sym] = ((dates.astype("int64").to_numpy() - eff) / 86_400_000_000_000).astype("float32")
    out["fund_age_days"] = age
    return out


if __name__ == "__main__":  # pragma: no cover
    r = build_records(D="data/curated")
    print("records", r.shape, "symbols", r.symbol.nunique())
    print("effective_date span:", r.effective_date.min(), "->", r.effective_date.max())
    print("non-null share:")
    print({c: round(float(r[c].notna().mean()), 2) for c in r.columns
           if c not in ("symbol", "effective_date", "period_end", "src")})
