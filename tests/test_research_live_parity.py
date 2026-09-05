"""The research path and the live path must compute the same thing.

THE BUG THIS PINS. Every number that justified the shipped model -- the prune,
the CPCV folds, both sealed holdouts -- came from
`validation.v3_panel.build_v3_panel`. The engine ranks with
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

from prosignal.features import v3_factors


def test_the_frame_length_is_a_single_constant():
    """Both callers read `FRAME_SESSIONS`. If either goes back to computing its
    own window, they can disagree again without anything failing."""
    assert v3_factors.FRAME_SESSIONS == v3_factors.LOOKBACK_SESSIONS + 16
    assert v3_factors.FRAME_SESSIONS == 316


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
        panel = sessions[max(i + 1 - v3_factors.FRAME_SESSIONS, 0): i + 1]
        live = cal.trailing_window(as_of, v3_factors.FRAME_SESSIONS)
        assert list(panel) == list(live), (
            f"the two paths disagree on the window ending {as_of}: "
            f"{len(panel)} rows vs {len(live)}")
        assert len(panel) == v3_factors.FRAME_SESSIONS


def test_the_window_covers_the_longest_factor():
    """`margin_stability` rolls 504 sessions but tolerates min_periods; the
    binding constraint the frame must satisfy is `prox_52w`'s 273."""
    assert v3_factors.FRAME_SESSIONS > 273
    assert v3_factors.LOOKBACK_SESSIONS >= 300
