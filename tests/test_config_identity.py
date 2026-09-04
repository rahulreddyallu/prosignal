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

import pytest

from prosignal.config import identity as ident
from prosignal.config.loader import load_config


class FakeStore:
    """A store with exactly the surface `identity` reads."""

    def __init__(self, *, sessions, price_symbols=100, delivery_symbols=90):
        import pandas as pd

        self._sessions = sessions
        self._px = pd.DataFrame({
            "date": pd.to_datetime(sessions * price_symbols)[:len(sessions) * price_symbols],
            "symbol": [f"S{i}" for i in range(price_symbols)
                       for _ in sessions][:len(sessions) * price_symbols],
            "close": 1.0,
        })
        self._dl = pd.DataFrame({
            "date": pd.to_datetime(sessions * delivery_symbols)[:len(sessions) * delivery_symbols],
            "symbol": [f"S{i}" for i in range(delivery_symbols)
                       for _ in sessions][:len(sessions) * delivery_symbols],
            "deliv_pct": 50.0,
        })

    def price_sessions(self):
        return list(self._sessions)

    def read_prices(self, *a, **k):
        return self._px

    def read_delivery(self, *a, **k):
        return self._dl

    def read_indices(self, *a, **k):
        return None

    def read_fundamentals(self, *a, **k):
        return None


def sessions(n, start=dt.date(2020, 1, 1)):
    return [start + dt.timedelta(days=i) for i in range(n)]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# -------------------------------------------------------------- the property
def test_a_deeper_store_changes_the_version_under_identical_parameters(cfg):
    """The hole, closed. This is the whole point of the module."""
    a = ident.identify(cfg, FakeStore(sessions=sessions(1000)))
    b = ident.identify(cfg, FakeStore(sessions=sessions(1250)))
    assert a.params_hash == b.params_hash, "the parameters did not move"
    assert a.combined != b.combined, (
        "250 extra sessions of price history left config_version identical. "
        "That is the defect: the model refits from the store, so those two "
        "runs are different models quoting the same identity.")


def test_a_wider_store_changes_the_version_too(cfg):
    """Symbol count, not only session count. Breadth moves coefficients."""
    a = ident.identify(cfg, FakeStore(sessions=sessions(500), price_symbols=100))
    b = ident.identify(cfg, FakeStore(sessions=sessions(500), price_symbols=140))
    assert a.combined != b.combined


def test_an_unchanged_store_gives_a_stable_version(cfg):
    """A version that churned for no reason is one nobody reads."""
    s = sessions(800)
    a = ident.identify(cfg, FakeStore(sessions=s))
    b = ident.identify(cfg, FakeStore(sessions=s))
    assert a.combined == b.combined
    assert a.version == b.version


def test_the_three_components_are_reported_separately(cfg):
    """XOR cannot say WHICH component moved, so `describe` has to."""
    i = ident.identify(cfg, FakeStore(sessions=sessions(600)))
    d = i.describe()
    for key in ("params_hash", "store_hash", "train_hash", "combined",
                "store_fingerprint", "train_window"):
        assert key in d and d[key], key
    assert d["params_hash"] != d["store_hash"] != d["train_hash"]


def test_the_combination_is_xor_of_the_three(cfg):
    i = ident.identify(cfg, FakeStore(sessions=sessions(600)))
    assert i.combined == ident.combine(i.params_hash, i.store_hash, i.train_hash)


# ------------------------------------------------------- the training window
def test_the_training_window_is_capped_and_the_cap_is_hashed(cfg):
    """The store can grow at the front without the training window moving."""
    cap = int(getattr(cfg.params.stage4_core_score,
                      "model_max_train_sessions", 3000))
    tw = ident.train_window(cfg, FakeStore(sessions=sessions(cap + 500)))
    assert tw.n_sessions == cap, (
        f"the training window reported {tw.n_sessions} sessions against a cap "
        f"of {cap}; the fit does not read more than the cap and the identity "
        f"must describe what is fitted.")


def test_the_window_carries_purge_and_embargo(cfg):
    """Changing either changes what is trained on and must move the hash."""
    tw = ident.train_window(cfg, FakeStore(sessions=sessions(500)))
    assert tw.purge_sessions >= tw.horizon_sessions, (
        "purge must be at least the label horizon; the loader enforces it and "
        "the identity records it.")
    a = tw.digest()
    b = dataclasses.replace(tw, embargo_sessions=tw.embargo_sessions + 1).digest()
    assert a != b


# ----------------------------------------------------------------- unknowns
def test_an_unreadable_feed_is_unknown_and_not_empty():
    """UNKNOWN and EMPTY produce different models; collapsing them hides a break."""

    class Broken(FakeStore):
        def read_delivery(self, *a, **k):
            raise OSError("parquet is corrupt")

    class Absent(FakeStore):
        def read_delivery(self, *a, **k):
            return None

    s = sessions(400)
    broken = ident.store_fingerprint(Broken(sessions=s))
    absent = ident.store_fingerprint(Absent(sessions=s))
    assert broken.digest() != absent.digest(), (
        "a feed that failed to read hashed the same as one that is genuinely "
        "absent. That lets a broken read masquerade as an empty feed, which is "
        "the failure the NOT_TESTABLE convention exists to prevent.")
    bad = next(f for f in broken.feeds if f.feed == "delivery")
    assert bad.unavailable and "corrupt" in bad.unavailable


# ------------------------------------------------------------- the AppConfig
def test_loading_a_config_does_not_require_a_store(cfg):
    """`config validate` and the schema tests have no data at all."""
    assert cfg.identity is None
    assert cfg.version == cfg.params_version
    assert cfg.version.endswith(cfg.hash)


def test_binding_a_store_moves_the_version_and_keeps_params_reachable(cfg):
    fresh = load_config(use_cache=False)
    before = fresh.version
    fresh.bind_store(FakeStore(sessions=sessions(900)))
    assert fresh.version != before
    assert fresh.params_version == before, (
        "`params_version` must stay the parameters-only answer: 'did the knobs "
        "move' is a real question and it is not the same question as 'is this "
        "the same model'.")
    assert fresh.identity is not None


def test_bind_store_returns_self_for_chaining(cfg):
    fresh = load_config(use_cache=False)
    assert fresh.bind_store(FakeStore(sessions=sessions(300))) is fresh


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
