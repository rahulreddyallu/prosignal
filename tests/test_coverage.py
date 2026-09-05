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
    """330 sessions: what the old bootstrap built to.

    The shortfall was 46 until prox_52w gained its 21-session
    reversal-avoiding offset. `model_minimum` is MIN_LOOKBACK + horizon + 60,
    and MIN_LOOKBACK moved 253 -> 274, so the bar moved 376 -> 397 and the same
    store is now 67 short rather than 46. The number is asserted rather than
    derived on purpose: it is the operator-facing figure, and a silent change
    to how much history the engine demands before it will fit at all is exactly
    the kind of thing this file exists to catch.
    """
    cov = assess(cfg, 330)
    assert cov.model_will_fit is False
    assert cov.ready is False
    assert cov.shortfall == 67
    assert "67 short" in cov.status()


def test_just_over_the_minimum_is_ready_but_says_it_is_not_validated(cfg):
    """Passing 397 means the fit will run, not that it is the fit that was
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


def test_the_interface_keeps_offering_the_button_until_the_store_is_deep_enough():
    """It disappeared at 330 because the screen keyed off `ready`, which was
    itself wrong.

    Both incomplete states still offer the build, but they no longer offer it
    the same way. Below the model minimum there is no ranking to withhold, so
    the build is the whole screen. Above it there ARE real picks, and hiding
    them behind a progress bar until the store reached nine years made a
    working engine look broken for months -- so the build becomes a tile over
    results that are shown with their caveat.
    """
    from pathlib import Path

    ui = (Path(__file__).resolve().parents[1] / "src" / "prosignal" / "static"
          / "index.html").read_text(encoding="utf-8")
    assert "matches_validation === false" in ui
    render = ui[ui.index("function render()"):ui.index("function buildTile")]
    assert "buildScreen()" in render, "the too-short state must reach the build"
    assert "buildTile()" in render, "the shallow state must still offer it"
    # The button lives on both, so it cannot vanish while history remains.
    assert ui.count('id="boot"') == 2


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
    """Keyed on the FAMILY columns, because that is what a fitted model emits.

    This fixture used to name individual factors -- `resid_mom`, `deliv_pct`,
    `prox_52w` -- which no code path has produced since the fit moved to
    families. The detection was keyed on the same stale names, so test and code
    agreed with each other while every healthy run was reported to the operator
    as 'the cross-sectional model could not fit this run ... treat this
    shortlist as unscored'. The warning was inverted, on every run.
    """
    from prosignal.features.crossmodel import FAMILY_COLUMNS, _bare
    from prosignal.presentation.viewmodel import _scorer_used

    got = _scorer_used([{"factors": {_bare(c): {} for c in FAMILY_COLUMNS}}])
    assert got["model"] == "cross-sectional", (
        "a run that carries the model's own family columns is still that "
        "model's run, and naming it tells a reader more than 'unknown'"
    )
    # `validated is True` and `note is None` USED TO BE ASSERTED HERE, and they
    # were right while the fitted model was live and evidenced. It was retired
    # on 2026-09-03. Certifying a retired model is what made V10's D-004
    # invisible: the interface renders its caveat only when `validated` is
    # false, so `True` meant the caveat could not appear, and `note: None` meant
    # there was nothing to render even if it had.
    #
    # Rendered from a stored pre-v3 payload on a current-dated store: "Ranked
    # from the close of <today>", five BUY badges, and no caveat anywhere.
    assert got["validated"] is False, (
        "nothing may certify a model that no longer exists"
    )
    assert got["note"], "and the reader has to be told which model it was"


def test_an_unrecognised_factor_block_reports_unknown_rather_than_guessing():
    """The failure direction has to be safe. Naming neither scorer is the
    honest answer; asserting either one is how a wrong claim ships."""
    from prosignal.presentation.viewmodel import _scorer_used

    got = _scorer_used([{"factors": {"something_the_engine_no_longer_emits": {}}}])
    assert got["model"] == "unknown"
    assert got["validated"] is False


def test_the_scorer_is_detected_from_the_factors_not_trusted_as_a_flag():
    """Nothing upstream sets a 'this was the fallback' field. Reading the
    factor names is the only signal that cannot be forgotten."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "prosignal"
           / "presentation" / "viewmodel.py").read_text(encoding="utf-8")
    fn = src[src.index("def _scorer_used"):src.index("def build_view")]
    assert "MODEL_KEYS" in fn and "COMPOSITE_KEYS" in fn


def test_the_interface_renders_the_unscored_warning():
    from pathlib import Path

    ui = (Path(__file__).resolve().parents[1] / "src" / "prosignal" / "static"
          / "index.html").read_text(encoding="utf-8")
    # The heading was "This shortlist is not from the model"; it is shorter
    # now. What matters is that the ALARM branch still renders and is still
    # keyed on the run not being validated.
    assert "Not ranked by the model" in ui
    assert 'sev === "alarm"' in ui


def test_the_empty_state_does_not_flash_before_the_data_arrives():
    """Boot renders once before it knows anything. Showing 'no scan yet' in
    that gap made every refresh flash the empty screen and replace it."""
    from pathlib import Path

    ui = (Path(__file__).resolve().parents[1] / "src" / "prosignal" / "static"
          / "index.html").read_text(encoding="utf-8")
    assert "state.booting" in ui
    render = ui[ui.index("function render()"):ui.index("function renderChrome")]
    assert "if (state.booting)" in render


def _ui() -> str:
    from pathlib import Path

    return (Path(__file__).resolve().parents[1] / "src" / "prosignal" / "static"
            / "index.html").read_text(encoding="utf-8")


def test_the_build_screen_shows_a_moving_session_count():
    """The job's own label is written once at the start and once at the end,
    so it sat unchanged for 24 minutes and made a working build look frozen."""
    ui = _ui()
    mon = ui[ui.index("function startMonitor"):ui.index("async function bootstrap")]
    assert "setInterval" in mon
    assert "clearInterval" in ui[ui.index("function stopMonitor"):], (
        "the monitor must stop when the server reports no job"
    )
    assert "price_sessions" in mon


def test_exactly_one_writer_owns_the_build_numbers():
    """Three things used to write numbers to that screen at once: a status
    line frozen at page load, the job's progress label reporting a figure
    minutes out of date, and the poller. They rendered together and
    disagreed -- 319 in one place and 369 in another, on the same screen.
    """
    ui = _ui()
    assert "function paintBuild" in ui
    mon = ui[ui.index("async function pollOnce"):ui.index("function setBuildIdle")]
    assert "paintBuild(c)" in mon
    assert "job.progress.label" not in ui, (
        "the job label is minutes stale and must not be rendered beside a "
        "live counter"
    )
    assert "boot-note" not in ui, "the second number's element is gone"


def test_the_counter_is_the_largest_thing_on_the_build_screen():
    """It is the answer to 'is this doing anything', so it is not a note under
    a button."""
    ui = _ui()
    css = ui[ui.index(".build .count .num"):]
    css = css[:css.index("}")]
    assert "clamp(38px" in css
    assert "tabular-nums" in css


def test_a_batched_jump_is_marked_so_it_reads_as_progress():
    """Sessions land in batches, so the number can sit still for two minutes
    and then jump 30. Without a cue that reads as a page reload."""
    ui = _ui()
    assert "tick" in ui[ui.index("async function bootstrap"):]


# ------------------------------------------------- long jobs survive the tab
def test_no_poll_loop_can_hang_on_a_terminal_state():
    """CANCELLED is terminal. A loop breaking only on COMPLETED and FAILED
    spins forever against a job that will never change again -- which is what
    produced an endless "Fetching" with a wall of HTTP 200s in the log and
    nothing wrong on the server.
    """
    ui = _ui()
    body = ui[ui.index("<script>"):]
    for block in body.split("for (;;)")[1:]:
        window = block[:1200]
        assert "CANCELLED" in window, (
            "a polling loop that does not handle CANCELLED can hang forever"
        )


def test_the_build_is_owned_by_the_server_not_by_a_view():
    """The loop used to belong to the bootstrap() call. Switching to History
    re-rendered the view and the build appeared to stop -- it had not, the
    server was still fetching, but nothing was watching it any more."""
    ui = _ui()
    assert "function startMonitor" in ui and "function stopMonitor" in ui
    boot = ui[ui.index("async function bootstrap"):ui.index("async function checkReady")]
    assert "for (;;)" not in boot, "bootstrap must not own a poll loop"
    assert "startMonitor()" in boot


def test_only_one_monitor_can_ever_run():
    """Two pollers would double the request rate and fight over the same DOM."""
    ui = _ui()
    fn = ui[ui.index("function startMonitor"):ui.index("function stopMonitor")]
    assert "if (monitorTimer) return" in fn


def test_a_reload_during_a_job_picks_it_back_up():
    """The server single-flights and keeps running regardless of the browser.
    A refresh that ignored that showed an idle button over a busy instance."""
    ui = _ui()
    boot = ui[ui.index("async function boot()"):]
    assert "if (state.jobId) startMonitor()" in boot
    screen = ui[ui.index("function buildScreen"):ui.index("function paintBuild")]
    assert "if (state.jobId)" in screen, (
        "re-entering the build screen mid-build must show the build"
    )


def test_the_screen_follows_the_job_kind_not_the_store_depth():
    """Reported live: pressing "Build data store" showed the nine-stage scan
    progress instead of the build.

    The monitor chose the screen with `if (!model_will_fit) build else scan`.
    That holds only while the store is too short. At 1,042 sessions the store
    DOES fit the model, so a bootstrap job satisfied the else branch and the
    build was rendered as a scan -- the deeper the store got, the more
    confidently it showed the wrong screen.

    The job carries its own kind. Nothing needs to be inferred.
    """
    ui = _ui()
    poll = ui[ui.index("async function pollOnce"):ui.index("function setBuildIdle")]
    assert 'kind === "bootstrap"' in poll
    assert "job.kind" in poll
    assert "if (!c.model_will_fit)" not in poll, (
        "store depth must not decide which job is running"
    )


def test_the_kind_is_known_without_asking_when_we_started_the_job():
    """A press that just POSTed a bootstrap does not need a round trip to
    learn what it started."""
    ui = _ui()
    boot = ui[ui.index("async function bootstrap"):ui.index("async function checkReady")]
    assert 'state.jobKind = "bootstrap"' in boot


def test_the_scan_button_is_hidden_on_the_build_screen():
    """renderChrome owns that button and the build screen returns before it,
    so the top "Scan Market" stayed visible over a store with nothing to
    scan -- two buttons, one of which could not work."""
    ui = _ui()
    assert "function hideScan" in ui
    render = ui[ui.index("function render()"):ui.index("function hideScan")]
    # The full build screen still hides it -- there is nothing to scan there.
    assert "hideScan(); return buildScreen();" in render
    # The tile does NOT, because those results are real and rescannable.
    assert "state.shallow" in render
