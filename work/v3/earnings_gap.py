"""Measure the earnings-window gap risk. Regenerates `features/earnings.py`'s
EARNINGS_RISK constants. TRAIN-FREE: this measures price behaviour around a
known calendar, not a return prediction, so it spends no holdout evidence.

THE COMPARISON HAS TO BE CONTROLLED AND THE OBVIOUS ONE IS NOT. Comparing
earnings sessions against "all sessions in the store" compares 179 large caps
with a calendar against 1,175 names including illiquid ones -- and flattered the
result badly the first time: it showed P(gap < -5%) at 1.30% against 0.82%, a
1.6x ratio. Restricted to the SAME names over the span their own calendar
covers, the true ratio is 4.94x. The naive version understates the risk by
three times, in the direction that makes holding through earnings look safe.
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/psr")
import data

D = "/mnt/user-data/uploads/Pro Stock Signal BOT/data/curated"
WINDOW_DAYS = 3
MIN_ANNOUNCEMENTS = 8


def main():
    px = data._adjust(data._load_prices())
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    e = pd.read_parquet(f"{D}/earnings_calendar.parquet")
    e["earnings_date"] = pd.to_datetime(e.earnings_date).dt.normalize()

    px = px.sort_values(["symbol", "date"])
    px["prev_close"] = px.groupby("symbol", observed=True)["close"].shift(1)
    px["gap"] = px["open"] / px["prev_close"] - 1.0
    px["day"] = px["close"] / px["prev_close"] - 1.0
    d = px.dropna(subset=["gap", "day"]).copy()

    have = e.groupby("symbol")["earnings_date"].agg(["min", "max", "size"])
    have = have[have["size"] >= MIN_ANNOUNCEMENTS]
    d = d[d.symbol.isin(have.index)].merge(have[["min", "max"]],
                                           left_on="symbol", right_index=True)
    d = d[(d.date >= d["min"]) & (d.date <= d["max"])].sort_values("date")

    ann = e[["symbol", "earnings_date"]].drop_duplicates().sort_values("earnings_date")
    j = pd.merge_asof(d[["symbol", "date"]], ann, left_on="date",
                      right_on="earnings_date", by="symbol", direction="backward")
    d = d.reset_index(drop=True)
    d["since"] = (j["date"] - j["earnings_date"]).dt.days.to_numpy()
    d["near"] = d["since"].between(0, WINDOW_DAYS)

    a, b = d.loc[d.near], d.loc[~d.near]
    out = {
        "window_days": WINDOW_DAYS,
        "symbols": int(d.symbol.nunique()),
        "sessions": int(len(d)),
        "sd_ratio": round(float(a.day.std() / b.day.std()), 2),
        "p_gap_below_5pct": round(float((a.gap < -0.05).mean()), 4),
        "p_gap_below_5pct_baseline": round(float((b.gap < -0.05).mean()), 4),
        "p_gap_below_5pct_ratio": round(float((a.gap < -0.05).mean() /
                                              (b.gap < -0.05).mean()), 2),
        "p_gap_below_8pct": round(float((a.gap < -0.08).mean()), 4),
        "p_gap_below_8pct_ratio": round(float((a.gap < -0.08).mean() /
                                              (b.gap < -0.08).mean()), 2),
        "session_p01": round(float(a.day.quantile(0.01)), 4),
        "session_p01_baseline": round(float(b.day.quantile(0.01)), 4),
    }
    for k, v in out.items():
        print(f"{k:32s} {v}")
    return out


if __name__ == "__main__":
    main()
