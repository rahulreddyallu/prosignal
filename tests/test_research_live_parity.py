"""The research path and the live path must compute the same thing.

THE BUG THIS PINS. Every number that justified the shipped model -- the prune,
the CPCV folds, both sealed holdouts -- came from
`validation.panel.build_panel`. The engine ranks with
`stage4_core_score.build_v3_block`. They share the factor formulae and the
scorer, so the arithmetic could not differ; what differed was the WINDOW. The
panel sliced `[i - LOOKBACK - 15 : i + 1]`, which is 316 rows inclusive, and the
live path asked the calendar for `LOOKBACK + 15`, which is 315.

One extra leading bar shifts the start of every rolling statistic. Measured on
six sampled dates the two scores agreed to Spearman 0.99997 and ranked an
identical top six -- close enough to pass any eyeball check, and different enough
that the model selected on the research path was not the model the engine ran.

That is the whole failure mode: not a wrong answer, a second answer.
"""

from __future__ import annotations

import datetime as dt

import pytest

from prosignal.features import v3_factors as factors


def test_the_frame_length_is_a_single_constant():
    """Both callers read `FRAME_SESSIONS`. If either goes back to computing its
    own window, they can disagree again without anything failing."""
    assert factors.FRAME_SESSIONS == factors.LOOKBACK_SESSIONS + 16
    assert factors.FRAME_SESSIONS == 436


def _code_lines(path):
    """Source with comments and docstrings stripped.

    The first version of the check below grepped raw text and failed on the
    COMMENT that explains the fix -- a test that forbids describing the bug it
    guards is a test nobody keeps.
    """
    import io, tokenize
    out = []
    with open(path, "rb") as fh:
        toks = list(tokenize.tokenize(fh.readline))
    prev_type = tokenize.INDENT
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and prev_type in (
                tokenize.INDENT, tokenize.NEWLINE, tokenize.NL, tokenize.DEDENT):
            continue          # a bare string statement: a docstring
        if tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                            tokenize.DEDENT):
            prev_type = tok.type
        out.append(tok.string)
    return " ".join(out)


def test_neither_caller_hardcodes_its_own_window():
    """The constant only helps while it is used, and the two expressions that
    drifted were both spelled `LOOKBACK + 15`. Checked against CODE, not text."""
    from pathlib import Path
    import prosignal
    root = Path(prosignal.__file__).resolve().parent
    for rel in ("stages/stage4_core_score.py", "validation/v3_panel.py"):
        code = _code_lines(root / rel)
        assert "FRAME_SESSIONS" in code, f"{rel} no longer reads the constant"
        assert "LOOKBACK_SESSIONS + 15" not in code, (
            f"{rel} computes its own frame length again")


def test_both_paths_select_the_same_sessions(monkeypatch):
    """The window each path would ask for, on a synthetic calendar, must be the
    same list -- which is what `max|diff| 0.0` in the parity harness rests on."""
    from prosignal.core.calendar import TradingCalendar
    import pandas as pd

    sessions = [d.date() for d in pd.bdate_range("2022-01-03", periods=900)]
    cal = TradingCalendar(sessions)
    for offset in (0, 7, 120):
        as_of = sessions[-1 - offset]
        i = sessions.index(as_of)
        panel = sessions[max(i + 1 - factors.FRAME_SESSIONS, 0): i + 1]
        live = cal.trailing_window(as_of, factors.FRAME_SESSIONS)
        assert list(panel) == list(live), (
            f"the two paths disagree on the window ending {as_of}: "
            f"{len(panel)} rows vs {len(live)}")
        assert len(panel) == factors.FRAME_SESSIONS


def test_the_window_covers_the_longest_factor():
    """`margin_stability` rolls 504 sessions but tolerates min_periods; the
    binding constraint the frame must satisfy is `prox_52w`'s 273."""
    assert factors.FRAME_SESSIONS > 273
    assert factors.LOOKBACK_SESSIONS >= 300


def test_both_paths_build_the_same_benchmark():
    """THE SECOND SPLIT, and the larger one. `resid_rev_21` is the only factor
    that reads a market return, and the two paths built different indices:

        panel  close.mean(axis=1).pct_change()      return of the MEAN PRICE
        live   (close.pct_change()).mean(axis=1)    MEAN OF RETURNS

    The first is price-weighted and dominated by whatever trades at four
    figures; the second is the equal-weight index `factor_frame`'s own contract
    promises. Raw `resid_rev_21` values differed by up to 1.30 on a live
    cross-section while the other twenty-one factors agreed exactly, so the
    factor was measured against one benchmark in every experiment and both
    sealed holdouts and computed against another in production.
    """
    import numpy as np
    import pandas as pd

    idx = pd.bdate_range("2024-01-01", periods=60)
    # Two names, one an order of magnitude more expensive: the whole point is
    # that a price-weighted index is not an equal-weighted one.
    close = pd.DataFrame({"CHEAP": np.linspace(10, 20, 60),
                          "DEAR": np.linspace(4000, 4100, 60)}, index=idx)
    price_weighted = (close.mean(axis=1) / close.mean(axis=1).shift(1) - 1.0)
    equal_weighted = (close / close.shift(1) - 1.0).mean(axis=1)
    assert not np.allclose(price_weighted.dropna(), equal_weighted.dropna()), (
        "the fixture must actually distinguish the two constructions")

    from pathlib import Path
    import prosignal
    root = Path(prosignal.__file__).resolve().parent
    # `_code_lines` joins tokens with spaces, so compare with whitespace removed
    # rather than against a source-formatted literal.
    want = "(close/close.shift(1)-1.0).mean(axis=1)"
    for rel in ("stages/stage4_core_score.py", "validation/v3_panel.py"):
        code = "".join(_code_lines(root / rel).split())
        assert want in code, (
            f"{rel} no longer builds the equal-weight benchmark")
        assert "close.mean(axis=1)/close.mean(axis=1).shift(1)" not in code, (
            f"{rel} is back to the return of the mean price")
