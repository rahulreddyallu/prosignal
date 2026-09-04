"""A config_version has to cover the DATA, not only the knobs.

THE HOLE. The model refits from stored history on every run, so the store IS
the training set. A store that grew produced different coefficients from
identical code and identical parameters -- under an identical `config_version`.
The README carried that as a known limitation while the forward test's entire
integrity check was "did `config_version` change", which is how the last window
recorded FIVE distinct model fingerprints against one config version.

    config_version = label @ ( H(params) XOR H(store) XOR H(train_window) )

These tests hold the property that matters -- two runs that trained on
different data cannot quote the same identity -- and the properties that keep
it usable: loading a config still works without a store, and re-verifying an
unchanged statutory rate does not churn the hash.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib

import pytest

from prosignal.config import identity as ident
from prosignal.config.loader import load_config


class FakeStore:
    """A store on disk, because the fingerprint reads parquet columns directly.

    It used to be a set of stub `read_*` methods. That stopped being a faithful
    double the moment `_coverage` was rewritten to read the parquet files --
    which it was, because going through `store.read_prices()` applied corporate
    actions to five million rows to count dates and symbols, and cost 10.8
    seconds on every CLI invocation. A double that does not exercise the real
    read path would have hidden that.
    """

    def __init__(self, root, *, sessions, price_symbols=100,
                 delivery_symbols=90):
        import pandas as pd

        self.curated = pathlib.Path(root)
        (self.curated / "prices").mkdir(parents=True, exist_ok=True)
        (self.curated / "delivery").mkdir(parents=True, exist_ok=True)
        self._sessions = list(sessions)

        def frame(n_syms):
            return pd.DataFrame({
                "date": [d for d in self._sessions for _ in range(n_syms)],
                "symbol": [f"S{i}" for _ in self._sessions
                           for i in range(n_syms)],
                "close": 1.0,
            })

        frame(price_symbols).to_parquet(
            self.curated / "prices" / "year=2020.parquet", index=False)
        frame(delivery_symbols).to_parquet(
            self.curated / "delivery" / "year=2020.parquet", index=False)

    def price_sessions(self):
        return list(self._sessions)


_N = [0]


def _u() -> str:
    """A fresh directory per store, so two stores in one test cannot collide."""
    _N[0] += 1
    return f"store{_N[0]}"


def sessions(n, start=dt.date(2020, 1, 1)):
    return [start + dt.timedelta(days=i) for i in range(n)]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# -------------------------------------------------------------- the property
def test_a_deeper_store_changes_the_version_under_identical_parameters(cfg, tmp_path):
    """The hole, closed. This is the whole point of the module."""
    a = ident.identify(cfg, FakeStore(tmp_path / _u(), sessions=sessions(1000)))
    b = ident.identify(cfg, FakeStore(tmp_path / _u(), sessions=sessions(1250)))
    assert a.params_hash == b.params_hash, "the parameters did not move"
    assert a.combined != b.combined, (
        "250 extra sessions of price history left config_version identical. "
        "That is the defect: the model refits from the store, so those two "
        "runs are different models quoting the same identity.")


def test_a_wider_store_changes_the_version_too(cfg, tmp_path):
    """Symbol count, not only session count. Breadth moves coefficients."""
    a = ident.identify(cfg, FakeStore(tmp_path / _u(), sessions=sessions(500), price_symbols=100))
    b = ident.identify(cfg, FakeStore(tmp_path / _u(), sessions=sessions(500), price_symbols=140))
    assert a.combined != b.combined


def test_an_unchanged_store_gives_a_stable_version(cfg, tmp_path):
    """A version that churned for no reason is one nobody reads."""
    s = sessions(800)
    root = tmp_path / "same"
    a = ident.identify(cfg, FakeStore(root, sessions=s))
    b = ident.identify(cfg, FakeStore(root, sessions=s))
    assert a.combined == b.combined
    assert a.version == b.version


def test_the_three_components_are_reported_separately(cfg, tmp_path):
    """XOR cannot say WHICH component moved, so `describe` has to."""
    i = ident.identify(cfg, FakeStore(tmp_path / _u(), sessions=sessions(600)))
    d = i.describe()
    for key in ("params_hash", "store_hash", "train_hash", "combined",
                "store_fingerprint", "train_window"):
        assert key in d and d[key], key
    assert d["params_hash"] != d["store_hash"] != d["train_hash"]


def test_the_combination_is_xor_of_the_three(cfg, tmp_path):
    i = ident.identify(cfg, FakeStore(tmp_path / _u(), sessions=sessions(600)))
    assert i.combined == ident.combine(i.params_hash, i.store_hash, i.train_hash)


# ------------------------------------------------------- the training window
def test_the_training_window_is_capped_and_the_cap_is_hashed(cfg, tmp_path):
    """The store can grow at the front without the training window moving."""
    cap = int(getattr(cfg.params.stage4_core_score,
                      "model_max_train_sessions", 3000))
    tw = ident.train_window(cfg, FakeStore(tmp_path / _u(), sessions=sessions(cap + 500)))
    assert tw.n_sessions == cap, (
        f"the training window reported {tw.n_sessions} sessions against a cap "
        f"of {cap}; the fit does not read more than the cap and the identity "
        f"must describe what is fitted.")


def test_the_window_carries_purge_and_embargo(cfg, tmp_path):
    """Changing either changes what is trained on and must move the hash."""
    tw = ident.train_window(cfg, FakeStore(tmp_path / _u(), sessions=sessions(500)))
    assert tw.purge_sessions >= tw.horizon_sessions, (
        "purge must be at least the label horizon; the loader enforces it and "
        "the identity records it.")
    a = tw.digest()
    b = dataclasses.replace(tw, embargo_sessions=tw.embargo_sessions + 1).digest()
    assert a != b


# ----------------------------------------------------------------- unknowns
def test_an_unreadable_feed_is_unknown_and_not_empty(tmp_path):
    """UNKNOWN and EMPTY produce different models; collapsing them hides a break."""

    s = sessions(400)
    # BROKEN: the file is present and cannot be read.
    broken_store = FakeStore(tmp_path / "broken", sessions=s)
    (broken_store.curated / "delivery" / "year=2020.parquet").write_bytes(
        b"this is not a parquet file")
    # ABSENT: the feed simply is not there.
    absent_store = FakeStore(tmp_path / "absent", sessions=s)
    for f in (absent_store.curated / "delivery").glob("*.parquet"):
        f.unlink()

    broken = ident.store_fingerprint(broken_store)
    absent = ident.store_fingerprint(absent_store)
    assert broken.digest() != absent.digest(), (
        "a feed that failed to read hashed the same as one that is genuinely "
        "absent. That lets a broken read masquerade as an empty feed, which is "
        "the failure the NOT_TESTABLE convention exists to prevent.")
    bad = next(f for f in broken.feeds if f.feed == "delivery")
    assert bad.unavailable, "a corrupt parquet was not flagged UNAVAILABLE"


# ------------------------------------------------------------- the AppConfig
def test_loading_a_config_does_not_require_a_store(cfg):
    """`config validate` and the schema tests have no data at all."""
    assert cfg.identity is None
    assert cfg.version == cfg.params_version
    assert cfg.version.endswith(cfg.hash)


def test_binding_a_store_moves_the_version_and_keeps_params_reachable(cfg, tmp_path):
    fresh = load_config(use_cache=False)
    before = fresh.version
    fresh.bind_store(FakeStore(tmp_path / _u(), sessions=sessions(900)))
    assert fresh.version != before
    assert fresh.params_version == before, (
        "`params_version` must stay the parameters-only answer: 'did the knobs "
        "move' is a real question and it is not the same question as 'is this "
        "the same model'.")
    assert fresh.identity is not None


def test_bind_store_returns_self_for_chaining(cfg, tmp_path):
    fresh = load_config(use_cache=False)
    assert fresh.bind_store(FakeStore(tmp_path / _u(), sessions=sessions(300))) is fresh


# ----------------------------------------------------------- hash discipline
def test_reverifying_an_unchanged_statutory_rate_does_not_move_the_hash():
    """Re-checking a rate that has not moved is not a change to the model.

    If the RATE moves, `value` moves and the hash moves with it, which is the
    behaviour that matters.
    """
    from prosignal.config.loader import config_hash

    a = load_config(use_cache=False)
    before = config_hash(a.params)
    node = a.params.costs.derivatives.stt_futures_sell_pct
    node.verified_on = dt.date(2027, 1, 1)
    assert config_hash(a.params) == before

    node.value = 0.07
    assert config_hash(a.params) != before, (
        "changing a statutory RATE left the config hash identical. The rate is "
        "an input to every net figure the engine produces.")
