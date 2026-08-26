"""The decay monitor, and the criterion it is not allowed to choose.

The gated estimator already zeroes a theme that cannot clear |t| >= 2, so a dead
theme stops being traded. That is a control, not a monitor: it acts and says
nothing about why, and a theme flickering in and out across refits looks
identical to one quietly dying. These tests pin the distinction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.validation.decay import (
    HAIRCUT_MCLEAN_PONTIFF, ThemeHealth, assess_decay)


def _panel(n_dates=70, n_names=60, betas=None, seed=0):
    """`betas` is the true slope per date, so decay can be constructed."""
    rng = np.random.default_rng(seed)
    if betas is None:
        betas = [0.5] * n_dates
    rows = []
    for d in range(n_dates):
        x = rng.normal(size=n_names)
        z = rng.normal(size=n_names)
        rows.append(pd.DataFrame({
            "date": pd.Timestamp("2018-01-01") + pd.Timedelta(days=21 * d),
            "symbol": [f"S{i}" for i in range(n_names)],
            "mom_f": x, "dead_f": z,
            "label_rank": betas[d] * x + rng.normal(size=n_names),
        }))
    return pd.concat(rows, ignore_index=True)


class TestItSeesDecay:
    def test_a_live_theme_is_kept(self):
        v = assess_decay(_panel(), ["mom_f", "dead_f"], window=24,
                         kill_t=0.0, required_breaches=24)
        mom = next(t for t in v.themes if t.theme == "mom_f")
        assert not mom.killed and mom.recent_t > 2

    def test_a_theme_that_died_halfway_is_killed(self):
        betas = [0.6] * 30 + [-0.05] * 40
        v = assess_decay(_panel(betas=betas, seed=5), ["mom_f", "dead_f"],
                         window=12, kill_t=0.0, required_breaches=12)
        mom = next(t for t in v.themes if t.theme == "mom_f")
        assert mom.killed, f"breaches={mom.breaches}, recent_t={mom.recent_t}"

    def test_one_bad_window_does_not_end_a_theme(self):
        """Breaching is not dying. That distinction is the whole reason the
        criterion requires a complete refresh of the window."""
        betas = [0.6] * 40 + [-0.1] * 4 + [0.6] * 26
        v = assess_decay(_panel(betas=betas, seed=6), ["mom_f"],
                         window=24, kill_t=0.0, required_breaches=24)
        assert not v.themes[0].killed

    def test_the_breach_run_is_counted_from_the_most_recent_check(self):
        betas = [-0.4] * 40 + [0.6] * 30
        v = assess_decay(_panel(betas=betas, seed=7), ["mom_f"],
                         window=12, kill_t=0.0, required_breaches=12)
        # It died and RECOVERED. A run counted from anywhere but the end would
        # kill a theme that is currently working.
        assert v.themes[0].breaches == 0 and not v.themes[0].killed


class TestTheCriterionIsPreCommitted:
    def test_it_ships_as_a_complete_window_refresh(self):
        import pathlib

        from prosignal.config.loader import load_config

        dm = load_config(
            pathlib.Path("config/parameters.yaml")
        ).params.stage4_core_score.decay_monitor
        assert dm.kill_t_stat == 0.0, "the bar is a sign test, not a tuned level"
        assert dm.required_breaches == dm.window_dates, (
            "the breach must persist across a COMPLETE refresh of the window, "
            "or a single bad quarter can end a theme"
        )

    def test_the_command_evaluates_the_config_and_does_not_choose(self):
        import inspect

        from prosignal import cli

        src = inspect.getsource(cli.cmd_research_decay)
        assert "dm.kill_t_stat" in src and "dm.required_breaches" in src
        for hardcoded in ("kill_t=0.0", "required_breaches=3", "window=24"):
            assert hardcoded not in src, f"{hardcoded} is not read from config"

    def test_an_unevaluable_criterion_kills_nothing(self):
        """Too short a panel must not be read as every theme passing, nor as
        every theme failing. It is refused, and said."""
        v = assess_decay(_panel(n_dates=20), ["mom_f"], window=24,
                         kill_t=0.0, required_breaches=24)
        assert v is not None and not v.killed
        assert any("not evaluable" in n for n in v.notes)


class TestTheHaircut:
    def test_the_expectation_is_haircut_not_the_full_estimate(self):
        v = assess_decay(_panel(), ["mom_f"], window=24, kill_t=0.0,
                         required_breaches=24, haircut=0.58)
        h = v.themes[0]
        assert h.expected_lambda == pytest.approx(h.full_lambda * 0.42)

    def test_the_default_haircut_is_mclean_pontiff(self):
        assert HAIRCUT_MCLEAN_PONTIFF == 0.58

    def test_a_share_of_a_meaningless_expectation_is_not_reported(self):
        """`reversal` printed "837% of expected" and `lottery` "4341%", both
        dividing by a coefficient that was itself noise, and both reading as
        though the theme were thriving."""
        h = ThemeHealth(theme="reversal_f", full_lambda=-0.0027, full_t=-0.22,
                        recent_lambda=-0.0096, recent_t=-0.50, recent_dates=24,
                        expected_lambda=-0.0027 * 0.42)
        assert np.isnan(h.share_of_expected)

    def test_a_measurable_expectation_is_reported(self):
        h = ThemeHealth(theme="mom_f", full_lambda=0.0704, full_t=3.35,
                        recent_lambda=0.0587, recent_t=2.63, recent_dates=24,
                        expected_lambda=0.0704 * 0.42)
        assert h.share_of_expected == pytest.approx(0.0587 / (0.0704 * 0.42))
