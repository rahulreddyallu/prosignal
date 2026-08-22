"""news_spike needs a move AND the volume that makes it an event.

The config declared move_sigma, volume_multiple and persistence_sessions with
the comment "Big move + big volume + NO scheduled event on the calendar =
unexplained". The code read move_sigma only, so the check fired on 1.30% of
stock-sessions against 1.01% with volume required -- 2.7x its own specification
once persistence is counted -- applying a 0.12 score penalty each time, which
reaches Stage 8's ordering and therefore capital.

volume_multiple is now read. persistence_sessions and
gap_signal.require_next_session_confirmation were REMOVED rather than
implemented, for reasons the tests below state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prosignal.config.loader import load_config
from prosignal.stages.stage5_false_signal import _news_spike


def _frame(n=120, spike_sigma=0.0, volume_multiple=1.0, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, n)
    if spike_sigma:
        r[-1] = spike_sigma * 0.01
    close = 100 * np.exp(np.cumsum(r))
    vol = np.full(n, 100_000.0)
    vol[-1] = 100_000.0 * volume_multiple
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": vol,
    })


@pytest.fixture
def cfg():
    return load_config().params.stage5_false_signal.news_spike


def test_a_big_move_on_big_volume_is_penalised(cfg):
    r = _news_spike(_frame(spike_sigma=5.0, volume_multiple=4.0), cfg)
    assert r.penalty > 0, "a 5-sigma move on 4x volume is the case this exists for"
    assert "volume" in (r.reason or "").lower()


def test_a_big_move_on_ordinary_volume_is_not_penalised(cfg):
    """The leg that was declared and never read. A large print in a thin book
    is not an event, and penalising it costs a name 0.12 for nothing."""
    r = _news_spike(_frame(spike_sigma=5.0, volume_multiple=1.0), cfg)
    assert r.penalty == 0, (
        "a sigma move without volume must not be penalised; reading move_sigma "
        "alone made this check fire 2.7x its specification"
    )


def test_an_ordinary_move_on_big_volume_is_not_penalised(cfg):
    r = _news_spike(_frame(spike_sigma=0.0, volume_multiple=5.0), cfg)
    assert r.penalty == 0


def test_missing_volume_reports_not_testable_rather_than_guessing(cfg):
    frame = _frame(spike_sigma=5.0)
    r = _news_spike(frame.drop(columns=["volume"]), cfg)
    assert r.penalty == 0
    assert "volume unavailable" in (r.reason or "").lower(), (
        "with no volume the check cannot distinguish an event from a thin "
        "print, and must say so rather than penalise or pass silently"
    )


def test_the_two_unimplementable_parameters_are_gone():
    """persistence_sessions contradicted the check's own rationale -- a move
    that persists is a trend, which is the opposite of what this penalises.
    require_next_session_confirmation needed tomorrow's data at today's
    decision. Neither was implemented; both were removed."""
    params = load_config().params.stage5_false_signal
    assert not hasattr(params.news_spike, "persistence_sessions")
    assert not hasattr(params.gap_signal, "require_next_session_confirmation")
