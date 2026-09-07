"""The book construction: breadth, full investment, and a buffer that buffers.

These are properties of the CONSTRUCTION, not claims about returns. That
distinction is the reason this file can exist at all right now: the trial
denominator behind the v3 ranking is unknown and unrecoverable
(docs/REBUILD_2026_09.md 3.1), so nothing here may assert that the resulting
book earns anything. What it asserts is that the book holds the names the
ranking nominates, in the size the specification says, with the capital
actually deployed -- which is exactly what the shipped sizer did not do.
"""
from __future__ import annotations

import pytest

from prosignal.book import BookSpec, build_book

N = 380
RANKING = [f"N{i:03d}" for i in range(N)]
PRICES = {t: 100.0 + i for i, t in enumerate(RANKING)}
DEEP = {t: 5e8 for t in RANKING}


@pytest.fixture
def spec():
    return BookSpec(capital=1_000_000.0)


# =============================================================================
# Full investment -- the defect this module replaces
# =============================================================================

def test_a_full_book_deploys_essentially_all_the_capital(spec):
    """THE POINT. `stage7_risk` invested 18.9% of capital on the same universe.

    Forgone benchmark return on the idle 81% at the eligible universe's ~21% a
    year is -17.0% annually, against a measured net excess of -17.6%. The cash
    drag was 97% of the underperformance, so this single assertion is worth
    more than every factor in the model.
    """
    b = build_book(RANKING, PRICES, DEEP, spec)
    assert b.invested_fraction > 0.99, (
        f"a full book deployed {b.invested_fraction:.1%} of capital"
    )
    assert b.invested_fraction <= 1.0, "the book cannot be levered"


def test_every_position_is_equal_weight_when_nothing_binds(spec):
    """1/N is the whole sizing rule. No stop distance, no risk category."""
    b = build_book(RANKING, PRICES, DEEP, spec)
    weights = [p.weight for p in b.positions]
    assert all(p.binding == "equal weight" for p in b.positions)
    # Whole-share rounding is the only source of spread, and it is tiny at
    # these prices; a wider spread means something else is sizing.
    assert max(weights) - min(weights) < 0.002, (
        f"weights span {min(weights):.4f}..{max(weights):.4f}; nothing should "
        f"differentiate them but share rounding"
    )


def test_breadth_is_a_decile_of_the_eligible_universe(spec):
    """Grinold: IR = IC x sqrt(breadth). Six names of 380 is not breadth."""
    b = build_book(RANKING, PRICES, DEEP, spec)
    assert len(b.positions) == 38 == int(N * spec.entry_percentile)
    assert [p.rank for p in b.positions] == list(range(1, 39)), (
        "the book must hold the TOP of the ranking, in order"
    )


def test_the_held_count_is_floored_and_capped(spec):
    """A thin day must not produce a three-name book."""
    small = RANKING[:60]
    b = build_book(small, PRICES, DEEP, spec)
    assert len(b.positions) == spec.min_names          # 6 would be the raw decile

    big = [f"B{i:04d}" for i in range(2000)]
    prices = {t: 100.0 for t in big}
    b2 = build_book(big, prices, {t: 5e8 for t in big}, spec)
    assert len(b2.positions) == spec.max_names


# =============================================================================
# The buffer band
# =============================================================================

def test_a_held_name_that_drifts_inside_the_wider_band_is_kept(spec):
    """Without a buffer a name oscillating on the boundary pays a full round
    trip at every rebalance for no change in view. This is how NSE, MSCI and
    FTSE construct factor indices, and the reason is arithmetic, not taste."""
    first = build_book(RANKING, PRICES, DEEP, spec)
    held = first.tickers

    # Push the top 20 down by 20 places. Names now at ranks 0-17 were held.
    drifted = RANKING[20:] + RANKING[:20]
    second = build_book(drifted, PRICES, DEEP, spec, held=held)

    kept = sum(1 for p in second.positions if p.held_before)
    assert kept == 18, f"expected the 18 survivors inside the band, kept {kept}"
    assert second.turnover < 0.6, (
        f"turnover {second.turnover:.0%} on a 20-place drift; the buffer is "
        f"not buffering"
    )
    assert second.invested_fraction > 0.99


def test_an_incumbent_that_drifts_past_the_target_count_still_keeps_its_slot(spec):
    """THE BUG THIS PINS, found 2026-09-07 by running the book on the real panel.

    The first implementation merged keepers and new names, sorted the union by
    rank and truncated to `n_target`. That looks equivalent to hysteresis and is
    not: `entry_cut` EQUALS `n_target`, so the top slots are always full of
    names ranked inside it, and every incumbent that drifted past `n_target` was
    truncated away however wide `exit_percentile` was set.

    The band was therefore inert, and inert in a way no synthetic test caught --
    `test_a_held_name_that_drifts_inside_the_wider_band_is_kept` drifts its
    holdings to ranks 0-17, which survive truncation for the wrong reason. On
    the real panel it showed up immediately: turnover was 62.3% per rebalance at
    an exit band of 20%, 30% AND 50%, identical to three significant figures.
    Fixed, the same measurement reads 35.3% / 23.9% / 14.0%.

    Here the whole held book drifts to ranks 39-76 -- outside the 38-name target
    and inside the 76-name band -- while 38 fresh names occupy the top. Under
    the old rule every incumbent was evicted. Under hysteresis a new name waits
    for a slot.
    """
    first = build_book(RANKING, PRICES, DEEP, spec)
    held = first.tickers
    assert len(held) == 38

    # Push the held block to ranks 39-76: past n_target, inside exit_cut (76).
    drifted = RANKING[38:76] + held + RANKING[76:]
    second = build_book(drifted, PRICES, DEEP, spec, held=held)

    assert all(p.held_before for p in second.positions), (
        "an incumbent inside the exit band was evicted by a fresh name; the "
        "buffer is inert and turnover will be roughly double what it should be"
    )
    assert second.turnover == 0.0
    assert second.invested_fraction > 0.99


def test_the_exit_band_width_actually_changes_what_is_held(spec):
    """A parameter that changes nothing is worse than no parameter."""
    first = build_book(RANKING, PRICES, DEEP, spec)
    held = first.tickers

    # Park the incumbents at ranks 151-188 of 380: outside an 11% band (41) and
    # comfortably inside a 90% one (342). Appending them to a slice leaves them
    # at the very bottom, which is outside BOTH bands and tests nothing.
    others = [t for t in RANKING if t not in set(held)]
    drifted = others[:150] + held + others[150:]

    narrow = build_book(drifted, PRICES, DEEP,
                        BookSpec(capital=1e6, exit_percentile=0.11), held=held)
    wide = build_book(drifted, PRICES, DEEP,
                      BookSpec(capital=1e6, exit_percentile=0.90), held=held)
    assert wide.turnover < narrow.turnover, (
        f"widening the band from 11% to 90% left turnover at "
        f"{narrow.turnover:.0%} -> {wide.turnover:.0%}"
    )


def test_a_name_outside_the_exit_band_is_sold(spec):
    first = build_book(RANKING, PRICES, DEEP, spec)
    held = first.tickers
    # Send the whole held set to the bottom.
    banished = RANKING[38:] + RANKING[:38]
    second = build_book(banished, PRICES, DEEP, spec, held=held)
    assert not any(p.held_before for p in second.positions)
    assert sorted(second.exits) == sorted(held)


def test_rebalancing_an_unchanged_ranking_trades_nothing(spec):
    """The cheapest possible rebalance is the one that does not happen."""
    first = build_book(RANKING, PRICES, DEEP, spec)
    second = build_book(RANKING, PRICES, DEEP, spec, held=first.tickers)
    assert second.turnover == 0.0
    assert second.entries == [] and second.exits == []


# =============================================================================
# Liquidity: cap the name, not the book
# =============================================================================

def test_a_binding_liquidity_cap_redistributes_rather_than_banking_cash(spec):
    """Capping and keeping the remainder is how a liquidity rule quietly
    becomes a market-timing rule -- a smaller version of the defect this
    module exists to remove."""
    thin = dict(DEEP)
    for t in RANKING[:10]:
        thin[t] = 5e5                       # Rs 5 lakh ADTV -> Rs 5,000 cap
    b = build_book(RANKING, PRICES, thin, spec)

    capped = [p for p in b.positions if p.binding == "liquidity cap"]
    assert len(capped) == 10
    assert all(p.value <= 5e5 * spec.max_participation_of_adtv + 1 for p in capped)
    assert b.invested_fraction > 0.99, (
        f"ten capped names left {1 - b.invested_fraction:.1%} of the book in "
        f"cash; the capital they could not take must go to the others"
    )


def test_a_name_with_unmeasured_liquidity_is_not_bought(spec):
    """An unmeasured quantity must reduce size, never license the largest one.
    `prosignal.liquidity` documents the original defect; this repeats its fix
    at book level rather than trusting the caller to have applied it."""
    adtv = dict(DEEP)
    adtv[RANKING[0]] = None                 # MISSING
    adtv[RANKING[1]] = 0.0                  # INVALID
    b = build_book(RANKING, PRICES, adtv, spec)

    assert RANKING[0] not in b.tickers and RANKING[1] not in b.tickers
    assert any("not sized" in n for n in b.notes)
    assert b.invested_fraction > 0.99, (
        "refusing two names must not leave their slots in cash"
    )


def test_cash_is_reported_when_it_is_genuinely_unavoidable(spec):
    """The one honest cash case: every held name is at its cap. It must be
    stated, not absorbed silently into a smaller-looking book."""
    thin = {t: 1e6 for t in RANKING}        # Rs 10,000 cap each, 38 names
    b = build_book(RANKING, PRICES, thin, spec)
    assert b.invested_fraction < 0.5
    assert any("could not be placed" in n for n in b.notes)


# =============================================================================
# Degenerate inputs
# =============================================================================

def test_an_empty_universe_yields_an_empty_book(spec):
    b = build_book([], PRICES, DEEP, spec)
    assert b.positions == [] and b.invested_fraction == 0.0
    assert b.notes


def test_a_name_without_a_price_is_not_sized(spec):
    prices = dict(PRICES)
    del prices[RANKING[0]]
    b = build_book(RANKING, prices, DEEP, spec)
    assert RANKING[0] not in b.tickers
    assert b.invested_fraction > 0.99
