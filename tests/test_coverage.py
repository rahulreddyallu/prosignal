"""How much stored history is enough -- answered once, not three times.

Three parts of the project used to answer this differently: /ready called the
store usable at 300 sessions, the bootstrap built to 330 and stopped, and the
ranking model abstains below 376. A fresh deployment built to 330, reported
itself ready, hid the build button and then produced no ranking at all. Every
number was defensible alone; the combination was a trap, and it was found by a
user watching a real deployment rather than by any test here.

The second failure is quieter and worse. The model REFITS from stored history
on every analysis, so the store is the training set. At 400 sessions it fits
on sixteen months where the shipped coefficients were validated on nine years
-- same code, same config hash, different model. The forward test's integrity
check would not catch it, because that hash covers parameters.yaml and not the
data underneath it.
"""

from __future__ import annotations

import pytest

from prosignal.config.loader import load_config
from prosignal.data.coverage import MINIMUM_NOTE, assess


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_the_model_minimum_is_derived_not_copied(cfg):
    """Duplicating the NUMBER would let it drift from the guard that enforces
    it. The expression is duplicated instead, so a change to either input
    moves both together."""
    from prosignal.features.crosssec import MIN_LOOKBACK
    from prosignal.stages._cfg import iv

    horizon = int(iv(cfg.params.stage4_core_score.model_horizon_sessions))
    assert assess(cfg, 0).model_minimum == MIN_LOOKBACK + horizon + 60


def test_the_bar_that_matters_is_the_models_not_eligibilitys(cfg):
    """min_history_sessions is when a STOCK becomes scoreable. It is not when
    the MODEL will fit, and using it as the readiness bar is what produced an
    instance that reported ready and ranked nothing."""
    cov = assess(cfg, 0)
    assert cov.model_minimum > cov.eligibility_minimum


def test_the_exact_store_that_broke_the_deployment_is_not_ready(cfg):
    """330 sessions: what the old bootstrap built to."""
    cov = assess(cfg, 330)
    assert cov.model_will_fit is False
    assert cov.ready is False
    assert cov.shortfall == 46
    assert "46 short" in cov.status()


def test_just_over_the_minimum_is_ready_but_says_it_is_not_validated(cfg):
    """Passing 376 means the fit will run, not that it is the fit that was
    measured. Serving that silently would let a short-history model pass for
    the validated one."""
    cov = assess(cfg, 400)
    assert cov.model_will_fit is True
    assert cov.ready is True
    assert cov.matches_validation is False
    assert "rather than" in cov.status()


def test_full_depth_reports_clean(cfg):
    cov = assess(cfg, 2210)
    assert cov.matches_validation is True
    assert "full validated depth" in cov.status()


def test_the_status_line_always_gives_a_number_to_act_on(cfg):
    for n in (0, 88, 330, 376, 900, 2200):
        assert any(ch.isdigit() for ch in assess(cfg, n).status())


def test_the_warning_says_why_a_short_store_is_not_merely_slower(cfg):
    """It is not a performance caveat. It is a different model."""
    assert "refits from stored history" in MINIMUM_NOTE
    assert "not the one the validation measured" in MINIMUM_NOTE


# ------------------------------------------------------------------ the API
def test_ready_reports_the_model_bar_not_the_eligibility_bar():
    from fastapi.testclient import TestClient

    from prosignal.api import create_app

    body = TestClient(create_app()).get("/ready").json()
    checks = body["checks"]
    assert "model_minimum" in checks
    assert "validated_target" in checks
    assert checks["model_minimum"] > checks["eligibility_minimum"]


def test_the_bootstrap_aims_at_the_validated_depth():
    """It used to stop at min_history_sessions + 30, which is below the point
    the model will even fit."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "prosignal"
           / "api.py").read_text(encoding="utf-8")
    runner = src[src.index("def _bootstrap_runner"):src.index('@app.post("/admin/bootstrap")')]
    assert "need = cov.validated_target" in runner
    assert "min_history_sessions.value) + 30" not in runner


def test_the_interface_keeps_offering_the_button_until_the_model_can_run():
    """It disappeared at 330 because the screen keyed off `ready`, which was
    itself wrong. It now also stays for a store that fits but is short of the
    validated depth."""
    from pathlib import Path

    ui = (Path(__file__).resolve().parents[1] / "src" / "prosignal" / "static"
          / "index.html").read_text(encoding="utf-8")
    assert "matches_validation === false" in ui
    assert "Continue building" in ui


def test_a_503_from_ready_is_read_not_discarded():
    """503 is the documented not-ready response and carries the body that
    explains why. Treating a non-2xx as unreadable threw away the numbers the
    screen needs to show."""
    from pathlib import Path

    ui = (Path(__file__).resolve().parents[1] / "src" / "prosignal" / "static"
          / "index.html").read_text(encoding="utf-8")
    block = ui[ui.index("async function checkReady"):]
    block = block[:block.index("\n}")]
    assert "r.json()" in block
    assert "if (!r.ok)" not in block, "a 503 body must still be parsed"


# ------------------------------------------------- which scorer actually ran
def test_a_composite_run_is_not_presented_as_a_model_run():
    """Stage 4 falls back to a hand-weighted composite when the
    cross-sectional model cannot fit, and treats an insufficient store as a
    benign reason -- so the run proceeds. The engine records that in a note.
    The note never reached the payload, so on a real deployment the composite
    rendered five confident BUY cards indistinguishable from the model's.

    The composite was measured at -0.047% excess per month against equal
    weight, t = -0.11.
    """
    from prosignal.presentation.viewmodel import _scorer_used

    # What the live server actually produced on 88 sessions.
    got = _scorer_used([{"factors": {"momentum_12_1": {},
                                     "sector_relative_strength": {}}}])
    assert got["model"] == "composite"
    assert got["validated"] is False
    assert "t = -0.11" in got["note"]


def test_a_real_model_run_is_not_flagged():
    from prosignal.presentation.viewmodel import _scorer_used

    got = _scorer_used([{"factors": {"resid_mom": {}, "deliv_pct": {},
                                     "prox_52w": {}}}])
    assert got["validated"] is True
    assert got["note"] is None


def test_the_scorer_is_detected_from_the_factors_not_trusted_as_a_flag():
    """Nothing upstream sets a 'this was the fallback' field. Reading the
    factor names is the only signal that cannot be forgotten."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "prosignal"
           / "presentation" / "viewmodel.py").read_text(encoding="utf-8")
    fn = src[src.index("def _scorer_used"):src.index("def build_view")]
    assert "FACTOR_MAP" in fn


def test_the_interface_renders_the_unscored_warning():
    from pathlib import Path

    ui = (Path(__file__).resolve().parents[1] / "src" / "prosignal" / "static"
          / "index.html").read_text(encoding="utf-8")
    assert "This shortlist is not from the model" in ui
    assert "scorer.validated === false" in ui


def test_the_empty_state_does_not_flash_before_the_data_arrives():
    """Boot renders once before it knows anything. Showing 'no scan yet' in
    that gap made every refresh flash the empty screen and replace it."""
    from pathlib import Path

    ui = (Path(__file__).resolve().parents[1] / "src" / "prosignal" / "static"
          / "index.html").read_text(encoding="utf-8")
    assert "state.booting" in ui
    render = ui[ui.index("function render()"):ui.index("function renderChrome")]
    assert "if (state.booting)" in render


def test_the_build_screen_shows_a_moving_session_count():
    """The job's own label is written once at the start and once at the end,
    so it sat unchanged for 24 minutes and made a working build look frozen."""
    from pathlib import Path

    ui = (Path(__file__).resolve().parents[1] / "src" / "prosignal" / "static"
          / "index.html").read_text(encoding="utf-8")
    block = ui[ui.index("async function bootstrap"):ui.index("async function checkReady")]
    assert "setInterval" in block
    assert "clearInterval" in block, "the poller must stop when the build ends"
    assert "price_sessions" in block
