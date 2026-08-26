"""The NO TRADE veto, and the reasons it is switched off.

Meta-labelling is a second model fitted only on the trades the primary would
have taken, predicting whether one reaches target before stop. It has no long
side; it can only refuse. These tests pin the construction, because the ways it
goes wrong are silent: fitted on an in-sample shortlist it approves everything,
and scored across dates it reports a discrimination it does not have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.features.metalabel import (
    MIN_META_ROWS, auc, fit_meta, fit_meta_out_of_sample, logistic_fit,
    logistic_predict, meta_label, reliability, shortlist)


def _rows(n=800, sep=1.2, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    win = rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(sep * x[:, 0] - 0.2)))
    side = np.where(win, 1.0, -1.0)
    return pd.DataFrame({
        "date": np.repeat(pd.date_range("2020-01-01", periods=n // 20, freq="21D"), 20),
        "symbol": [f"S{i}" for i in range(n)],
        "mom_f": x[:, 0], "lottery_f": x[:, 1],
        "barrier_side": side, "uniqueness": 0.6,
        "label": np.where(win, 0.10, -0.06),
    })


class TestLogistic:
    def test_it_recovers_the_coefficients_it_was_given(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(6000, 3))
        eta = 1.2 * x[:, 0] - 0.8 * x[:, 1] - 0.3
        y = (rng.uniform(size=6000) < 1 / (1 + np.exp(-eta))).astype(float)
        f = logistic_fit(x, y, l2=1.0)
        assert f["converged"]
        assert f["coef"][0] == pytest.approx(1.2, abs=0.15)
        assert f["coef"][1] == pytest.approx(-0.8, abs=0.15)
        assert abs(f["coef"][2]) < 0.15

    def test_probabilities_stay_inside_zero_and_one(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(500, 2)) * 50.0        # drives eta to saturation
        y = (x[:, 0] > 0).astype(float)
        p = logistic_predict(logistic_fit(x, y), x)
        assert np.isfinite(p).all() and p.min() >= 0.0 and p.max() <= 1.0

    def test_the_intercept_is_not_penalised(self):
        """A penalised intercept drags every prediction toward 0.5 regardless of
        the base rate, which is a different claim entirely."""
        rng = np.random.default_rng(3)
        x = rng.normal(size=(3000, 2))
        y = (rng.uniform(size=3000) < 0.15).astype(float)     # rare event
        f = logistic_fit(x, y, l2=500.0)
        assert logistic_predict(f, x).mean() == pytest.approx(0.15, abs=0.03)


class TestTheLabel:
    def test_target_is_one_stop_is_zero_timeout_is_neither(self):
        out = meta_label(pd.Series([1.0, -1.0, 0.0, np.nan]))
        assert out.iloc[0] == 1.0 and out.iloc[1] == 0.0
        assert np.isnan(out.iloc[2]) and np.isnan(out.iloc[3])

    def test_a_timeout_is_excluded_not_counted_as_a_loss(self):
        rows = _rows(400)
        rows.loc[rows.index[:100], "barrier_side"] = 0.0
        model, why = fit_meta(rows, ["mom_f", "lottery_f"], min_rows=50)
        assert model is not None, why
        assert model.n_train == 300


class TestFitting:
    def test_it_learns_a_separation_that_is_really_there(self):
        model, why = fit_meta(_rows(sep=1.5), ["mom_f", "lottery_f"], min_rows=100)
        assert model is not None, why
        assert auc(meta_label(_rows(sep=1.5)["barrier_side"]).to_numpy(),
                   model.predict_proba(_rows(sep=1.5))) > 0.65

    def test_it_refuses_rather_than_fitting_a_handful_of_trades(self):
        model, why = fit_meta(_rows(120), ["mom_f"], min_rows=MIN_META_ROWS)
        assert model is None and "required" in why

    def test_it_refuses_when_every_trade_had_the_same_outcome(self):
        rows = _rows(600)
        rows["barrier_side"] = 1.0
        model, why = fit_meta(rows, ["mom_f"], min_rows=50)
        assert model is None and "same outcome" in why

    def test_it_refuses_without_a_barrier_outcome_to_learn_from(self):
        rows = _rows(600).drop(columns=["barrier_side"])
        model, why = fit_meta(rows, ["mom_f"], min_rows=50)
        assert model is None and "triple-barrier" in why


class TestTheShortlistIsOutOfSample:
    def test_the_primary_never_ranks_the_rows_the_veto_learns_from(self):
        """Shortlisting with an in-sample primary selects the names it already
        got right, and a classifier fitted on those learns to approve
        everything."""
        rows = _rows(1600)
        seen: list = []

        def primary(train, frame):
            seen.append((set(train["date"].unique()), set(frame["date"].unique())))
            return frame["mom_f"].to_numpy("float64")

        model, why = fit_meta_out_of_sample(
            rows, ["mom_f", "lottery_f", "_meta_score"], primary,
            top_k=10, min_rows=50)
        assert model is not None, why
        train_dates, score_dates = seen[0]
        assert not (train_dates & score_dates), "the primary scored its own fit block"

    def test_it_refuses_when_there_are_too_few_dates_to_split(self):
        rows = _rows(100)                       # 5 dates, 20 names each
        model, why = fit_meta_out_of_sample(rows, ["mom_f"], lambda t, f: None,
                                            min_rows=10)
        assert model is None and "out of sample" in why

    def test_a_primary_that_cannot_fit_is_reported_not_swallowed(self):
        model, why = fit_meta_out_of_sample(_rows(1600), ["mom_f"],
                                            lambda t, f: None, min_rows=50)
        assert model is None and "primary" in why

    def test_the_shortlist_keeps_the_top_k_per_date(self):
        rows = _rows(800)
        out = shortlist(rows, rows["mom_f"].to_numpy(), top_k=5)
        assert (out.groupby("date").size() <= 5).all()
        assert "_meta_score" in out.columns


class TestMeasurement:
    def test_auc_is_a_half_for_a_coin_and_one_for_perfect(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
        assert auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)

    def test_auc_is_undefined_with_only_one_class(self):
        assert np.isnan(auc(np.ones(10), np.linspace(0, 1, 10)))

    def test_reliability_compares_predicted_against_realised(self):
        rng = np.random.default_rng(5)
        p = rng.uniform(size=4000)
        y = (rng.uniform(size=4000) < p).astype(float)
        table = reliability(y, p, bins=4)
        assert len(table) == 4
        assert (table["realised"] - table["predicted"]).abs().max() < 0.08


class TestTheGateIsOffAndSaysSo:
    def test_the_shipped_config_vetoes_nobody(self):
        import pathlib

        from prosignal.config.loader import load_config

        cfg = load_config(pathlib.Path("config/parameters.yaml"))
        assert cfg.params.stage4_core_score.metalabel.enabled is False
        assert float(
            cfg.params.stage8_final_signal.scarcity.min_win_probability.value) == 0.0

    def test_an_unscored_name_is_not_treated_as_approved(self):
        """A gap is not a pass. This is how a gate quietly stops gating."""
        import inspect

        from prosignal.stages import stage8_final_signal

        src = inspect.getsource(stage8_final_signal.run)
        assert "prob is None or prob < min_win" in src

    def test_held_positions_are_exempt_from_the_veto(self):
        import inspect

        from prosignal.stages import stage8_final_signal

        src = inspect.getsource(stage8_final_signal.run)
        assert "sym not in open_book" in src


class TestTheVetoTravelsWithTheModel:
    """The cheap path scores from a CACHED model on 20 of every 21 sessions.

    A classifier that is not cached is absent on those days, every candidate
    scores as unknown, and a gate whose rule is "unknown is not approved"
    silently refuses the entire book. Caught by running it: buys fell from 8 to
    4 -- the 4 being held positions, which are exempt -- with no error anywhere.
    """

    def _model(self, tmp_path):
        import datetime as dt

        import numpy as np

        from prosignal.features import crossmodel as cm
        from prosignal.features.metalabel import MetaModel

        m = cm.CrossSectionalModel(
            coef={"mom_f": 0.06, "delivery_f": 0.04}, n_train=5000,
            train_end=dt.date(2026, 1, 5), features=["mom_f", "delivery_f"])
        m.mu = np.array([0.0, 0.0])
        m.sd = np.array([1.0, 1.0])
        m.intercept = 0.0
        m.estimator = "fama_macbeth"
        m.meta = MetaModel(
            features=["mom_f", "delivery_f", "_meta_score"],
            coef=np.array([0.5, 0.2, 0.1]), intercept=-0.1,
            mu=np.array([0.0, 0.0, 0.0]), sd=np.array([1.0, 1.0, 1.0]),
            n_train=800, base_rate=0.47)
        return m

    def test_a_cached_model_carries_its_veto(self, tmp_path):
        import datetime as dt

        from prosignal.features import crossmodel as cm

        path = tmp_path / "model.json"
        cm.save_cache(path, self._model(tmp_path), dt.date(2026, 1, 6))
        back = cm.load_cached(path, dt.date(2026, 1, 6), 21,
                              estimator="fama_macbeth")
        assert back is not None and back.meta is not None
        assert back.meta.features == ["mom_f", "delivery_f", "_meta_score"]
        assert back.meta.base_rate == pytest.approx(0.47)

    def test_the_cheap_scoring_path_produces_probabilities(self, tmp_path):
        import datetime as dt

        from prosignal.features import crossmodel as cm

        path = tmp_path / "model.json"
        cm.save_cache(path, self._model(tmp_path), dt.date(2026, 1, 6))
        back = cm.load_cached(path, dt.date(2026, 1, 6), 21,
                              estimator="fama_macbeth")
        frame = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"],
                              "mom_f": [0.8, 0.0, -0.8],
                              "delivery_f": [0.5, 0.0, -0.5]})
        cm.score_with(back, frame)
        assert back.meta_prob is not None and len(back.meta_prob) == 3
        assert back.meta_prob["AAA"] > back.meta_prob["CCC"]

    def test_a_model_without_a_veto_round_trips_as_none(self, tmp_path):
        import datetime as dt

        from prosignal.features import crossmodel as cm

        m = self._model(tmp_path)
        m.meta = None
        path = tmp_path / "model.json"
        cm.save_cache(path, m, dt.date(2026, 1, 6))
        back = cm.load_cached(path, dt.date(2026, 1, 6), 21,
                              estimator="fama_macbeth")
        assert back is not None and back.meta is None


class TestTheFunnelStaysMonotonic:
    def test_the_pipeline_uses_stage_eights_own_counts(self):
        """It rebuilt them by hand on the trade path, reading
        `entries.triggered()` -- the population BEFORE the score gate -- so the
        funnel could run backwards on exactly the path that produces a trade."""
        import inspect

        from prosignal import pipeline

        src = inspect.getsource(pipeline._run_analysis_locked)
        assert "funnel = no_trade.gate_summary if no_trade else gate_counts" in src
        assert "len(entries.triggered())" not in src

    def test_stage_eight_returns_its_counts_on_every_path(self):
        """Parsed, not string-matched: every exit of `run` must hand back a
        4-tuple ending in `gate_counts`. A path that forgets it leaves the
        pipeline with nothing to render but the hand-built funnel again."""
        import ast
        import inspect
        import textwrap

        from prosignal.stages import stage8_final_signal

        tree = ast.parse(textwrap.dedent(
            inspect.getsource(stage8_final_signal.run)))
        fn = tree.body[0]
        nested = {n for d in ast.walk(fn)
                  if isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef)) and d is not fn
                  for n in ast.walk(d)}
        exits = [n for n in ast.walk(fn)
                 if isinstance(n, ast.Return) and n not in nested]
        assert len(exits) >= 4, f"expected every exit path, found {len(exits)}"
        for node in exits:
            assert isinstance(node.value, ast.Tuple), ast.unparse(node)
            assert len(node.value.elts) == 4, ast.unparse(node)
            last = node.value.elts[-1]
            assert isinstance(last, ast.Name) and last.id == "gate_counts", \
                ast.unparse(node)

    def test_the_veto_step_sits_between_the_score_gate_and_the_trigger(self):
        """A key appended after the dict is built lands at the BOTTOM of the
        funnel, below gates it actually precedes."""
        import inspect

        from prosignal.stages import stage8_final_signal

        src = inspect.getsource(stage8_final_signal.run)
        i = src.index('"passed_score_threshold": 0')
        j = src.index('gate_counts["passed_meta_label"] = 0')
        k = src.index('gate_counts["triggered"] = 0')
        assert i < j < k
