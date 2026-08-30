"""One place that decides what may be used as a factor.

WHY THIS EXISTS. The screen picked its columns as "everything that is not
metadata", and the panel carries `mae5` and `mfe5` -- the worst and best price
reached in the FIVE SESSIONS AFTER the signal. They are there so a stop-loss can
be simulated. They are pure lookahead, and the screen duly reported rank ICs of
+0.22 and +0.25 with t-statistics above 30, which is what a factor that has read
the future looks like.

Nothing consumed them: the composite takes its columns from `themes.THEMES`, and
neither is in it. But "nothing consumed it" is luck, not a control. Every
factor list now comes through `factor_columns`, which takes the intersection of
the panel with the declared themes and refuses anything on the forbidden list.
"""
from __future__ import annotations
import themes as TH

#: Columns computed from sessions AFTER the signal date. Never a factor.
FORWARD_LOOKING = frozenset({
    "mae5", "mfe5", "entry_px",
} | {f"y{h}" for h in (5, 10, 21, 42, 63, 126)}
  | {f"b{h}" for h in (5, 10, 21, 42, 63, 126)})

#: Columns that describe the row rather than the company.
BOOKKEEPING = frozenset({"date", "symbol", "sector", "adtv", "adtv_rank",
                         "close", "atr_pct", "mcap", "fund_age_days"})


class LookaheadError(AssertionError):
    pass


def factor_columns(df, themes=None):
    """Every declared factor the frame actually carries, and nothing else."""
    want = [f for f in TH.FACTOR_THEME
            if (themes is None or TH.FACTOR_THEME[f] in themes)]
    cols = [c for c in want if c in df.columns]
    bad = FORWARD_LOOKING & set(cols)
    if bad:
        raise LookaheadError(
            f"{sorted(bad)} are computed from sessions after the signal date "
            f"and cannot be factors. They are in the panel so a stop can be "
            f"simulated; a screen that picks columns by exclusion will find "
            f"them and report an information coefficient above 0.2.")
    return cols


def assert_no_lookahead(cols):
    bad = FORWARD_LOOKING & set(cols)
    if bad:
        raise LookaheadError(f"forward-looking columns in a factor list: {sorted(bad)}")
    return True
