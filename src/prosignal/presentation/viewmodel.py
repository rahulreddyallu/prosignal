"""One business-ready shape for the whole screen.

The interface binds to this and never to the engine's own output. That
boundary is the point: the raw payload carries factor coefficients, stage
numbers, standardised loadings and rule identifiers, and every one of those
leaking into the interface is how the old screen ended up reading like a
console. The raw endpoint stays available for the research view -- the
technical detail is not deleted, it is moved behind a door.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

from ..core.logging import get_logger
from .evidence import build_evidence, category_summary, confirmation_count
from .narrative import build_narrative
from .selection import BUY, SLOTS, WATCH, select_slate

log = get_logger(__name__)

#: Engine vocabulary -> the words the interface uses. The funnel keys are the
#: engine's own and would otherwise reach the screen verbatim.
FUNNEL_LABELS: Dict[str, str] = {
    "universe_considered": "Stocks evaluated",
    "passed_eligibility": "Passed data and liquidity checks",
    "scored": "Ranked by the model",
    "survived_defense": "Cleared the risk checks",
    "triggered": "Met the entry criteria",
    "passed_meta_label": "Cleared the NO TRADE veto",
    "buys": "Qualifying setups",
}

FUNNEL_DETAIL: Dict[str, str] = {
    "universe_considered": "Every listed name the run started from.",
    "passed_eligibility": (
        "Names with enough trading history, sufficient liquidity to enter and "
        "exit, and no unexplained price discontinuity."
    ),
    "scored": (
        "Each surviving name ranked against every other on the same day, so a "
        "reading is always relative to the market rather than to a fixed level."
    ),
    "survived_defense": (
        "Checks for gaps, news spikes, overextension, corporate actions and "
        "earnings distortion. A name failing one is penalised or removed."
    ),
    "triggered": "Names inside the rank band the strategy opens positions in.",
    "passed_meta_label": (
        "New entries a second model did not refuse. It estimates the chance a "
        "trade like this reaches its target before its stop; anything below the "
        "floor is dropped. It can only refuse, never propose. Absent from this "
        "funnel when the veto is switched off, which is the shipped default."
    ),
    "buys": "What remained after every check.",
}

_STRENGTH_LABEL = {"High": "Strong", "Medium": "Moderate", "Low": "Emerging"}


def _company(card: Dict[str, Any], names: Dict[str, str]) -> str:
    """A readable name, falling back to the ticker rather than to nothing.

    The engine leaves `company_name` unset on every row -- 0 of 52 on a live
    run -- while the store holds 2,565 of them in the equity master. The join
    happens here so the interface never has to.
    """
    existing = card.get("company_name")
    if existing:
        return str(existing)
    ticker = str(card.get("ticker") or "")
    return names.get(ticker) or ticker


def _tidy_company(name: str) -> str:
    """NSE stores these in mixed case with a legal suffix. Drop the suffix and
    leave the case alone -- guessing at capitalisation mangles acronyms."""
    for suffix in (" Limited", " Ltd.", " Ltd", " LIMITED"):
        if name.endswith(suffix):
            return name[: -len(suffix)].strip()
    return name


def _scorer_used(picks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Which scorer actually produced this ranking.

    Stage 4 falls back to a hand-weighted composite when the cross-sectional
    model cannot fit -- an insufficient store is treated as a benign reason, so
    the run proceeds rather than failing. That is a defensible degraded mode
    and the engine records it in a note. The note never reached this payload,
    so the composite's output rendered identically to the model's: same cards,
    same green contributions, same BUY.

    The composite was measured at -0.047%/month excess against an equal-weight
    benchmark, t = -0.11. Presenting it as the validated model is the single
    most misleading thing this interface could do, so it is detected here from
    the factor names themselves rather than trusted to arrive as a flag.

    DETECTED BY THE COMPOSITE'S OWN KEYS, not by failing to recognise the
    model's. This read `seen & FACTOR_MAP` while FACTOR_MAP still held the
    pre-family factor names, so the fitted model's family keys matched nothing
    and every healthy run was reported as "the cross-sectional model could not
    fit this run ... treat this shortlist as unscored". That is the exact
    misrepresentation this function exists to prevent, inverted. Keying on the
    composite's four names makes the failure direction safe: an unrecognised
    key set reports UNKNOWN rather than asserting either scorer.
    """
    from .evidence import COMPOSITE_KEYS, MODEL_KEYS

    seen: set = set()
    for pick in picks:
        seen.update((pick.get("factors") or {}).keys())
    if not seen:
        return {"model": "unknown", "validated": False,
                "note": "No factor detail was recorded for this run."}
    # Decided on the keys only ONE scorer can emit. `value` and `quality` are
    # both fitted families and hand-weighted composite factors -- the engine
    # genuinely uses the same two words for both -- so a shared key identifies
    # nothing and matching on it would call a composite run a model run.
    if seen & (MODEL_KEYS - COMPOSITE_KEYS):
        return {"model": "cross-sectional", "validated": True, "note": None}
    if not (seen & (COMPOSITE_KEYS - MODEL_KEYS)):
        return {
            "model": "unknown",
            "validated": False,
            "factors": sorted(seen),
            "note": (
                "This run's factor names match neither the fitted model nor the "
                "hand-weighted composite, so which scorer produced the ranking "
                "cannot be established from the payload. Treat the shortlist as "
                "unattributed until that is resolved."
            ),
        }
    return {
        "model": "composite",
        "validated": False,
        "factors": sorted(seen),
        "note": (
            "The cross-sectional model could not fit this run, so the ranking "
            "came from the hand-weighted composite instead. That composite was "
            "measured at -0.047% excess per month against an equal-weight "
            "benchmark, t = -0.11 -- it is a placeholder, not a signal. Treat "
            "this shortlist as unscored."
        ),
    }


def build_view(
    payload: Dict[str, Any],
    *,
    company_names: Optional[Dict[str, str]] = None,
    sectors: Optional[Dict[str, str]] = None,
    entry_rank: int = 8,
    exit_rank: int = 16,
    slots: int = SLOTS,
    horizon_sessions: Optional[int] = None,
    entry_clock: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the screen's data from one completed analysis payload."""
    names = company_names or {}
    sector_map = sectors or {}

    recommendations = list(payload.get("recommendations") or [])
    watchlist = list(payload.get("watchlist") or [])

    flags = list(payload.get("data_quality_flags") or [])
    # The run decided the screen and recorded it. Rendering it is all that is
    # left to do here. Recomputing it would reintroduce exactly the divergence
    # this field exists to close -- a reader cannot see the previous screen, so
    # anything it computes is a fresh top-N with no memory, whatever the engine
    # actually held.
    slate = _recorded_slate(payload, recommendations, watchlist, slots=slots)

    # Only the slate is shipped. The full ranked watchlist had its own screen
    # and no longer does -- serialising forty-odd extra names that nothing
    # renders is weight on every response for no reader.
    picks = [
        _build_pick(card, names, sector_map,
                    entry_rank=entry_rank, exit_rank=exit_rank)
        for card in slate.picks
    ]

    return {
        "as_of": payload.get("as_of_date"),
        "generated_at": payload.get("generated_at"),
        "run_id": payload.get("run_id"),
        "market": _market(payload.get("regime") or {}),
        "summary": {
            "buy": slate.buy_count,
            "watch": slate.watch_count,
            "total": len(picks),
            "note": slate.selection_note,
            "withheld": slate.withheld_reason,
        },
        "picks": picks,
        # THE SHAPE OF THE BOOK, so the screen stops carrying its own copy of
        # it. The interface had "top 18", the horizon and the book size written
        # into strings in three places, which is how it went on describing a
        # five-name book with a profit target after the engine had moved to six
        # names with neither.
        "book": {
            "size": int(slots),
            "entry_rank": int(entry_rank),
            "exit_rank": int(exit_rank),
            "horizon_sessions": horizon_sessions,
        },
        # WHAT THE ENGINE ESTIMATES AND WHAT IT DOES NOT, once, for the whole
        # list. Every return and hold figure this engine has is a POPULATION
        # base rate from one study -- the same 7.09% and the same 42 sessions
        # on every name -- so a per-row "expected return" column would be the
        # same number repeated down the page while reading as a per-name
        # forecast. There is no per-name return estimate to show. Stating that
        # once, next to the study figures, is the honest version of the column
        # that keeps being looked for.
        "expectancy": _expectancy(recommendations + watchlist),
        "departures": list(payload.get("slate_departures") or []),
        "new_entries_blocked": payload.get("new_entries_blocked"),
        "entry_clock": dict(entry_clock or payload.get("entry_clock") or {}),
        # Held names the run could not evaluate: suspended, dropped from the
        # universe, or delisted. This is the risk the screen was least able to
        # show, because the position simply stopped being mentioned.
        "open_position_alerts": _position_alerts(payload),
        "scorer": _scorer_used(recommendations + watchlist),
        "concentration": _concentration(picks),
        "journey": _journey(payload.get("funnel") or {}, slate),
        "data": _data_state(payload, flags),
        "disclaimer": payload.get("disclaimer"),
        "confidence_note": payload.get("probability_note"),
    }


def _recorded_slate(payload, recommendations, watchlist, *, slots):
    """Render the slate the run recorded; compute one only if it did not.

    The fallback is for payloads produced before the slate became part of the
    run record. It is deliberately the old stateless top-N -- there is no
    previous screen available at read time, so pretending to apply the hold
    band here would produce a different list every time it was called and
    label it "held".
    """
    from .selection import Slate

    recorded = list(payload.get("slate") or [])
    if not recorded:
        return select_slate(recommendations, watchlist, slots=slots)

    by_ticker = {}
    for card in list(recommendations) + list(watchlist):
        by_ticker.setdefault(str(card.get("ticker")), card)

    picks = []
    n_buy = n_watch = carried = 0
    for entry in recorded:
        ticker = str(entry.get("ticker") or "")
        card = by_ticker.get(ticker)
        if card is None:
            # The run named a row its own payload does not carry. Dropping it
            # is right -- there is nothing to render -- but it is a contract
            # breach, not a normal state, so it must not pass unremarked.
            log.warning("slate names a ticker absent from the payload",
                        extra={"ticker": ticker})
            continue
        row = dict(card)
        row["slate_position"] = entry.get("position")
        row["status"] = entry.get("status") or WATCH
        row["carried"] = bool(entry.get("carried"))
        row["slate_reason"] = entry.get("reason") or ""
        row["shown_since"] = entry.get("shown_since")
        if row["status"] == BUY:
            n_buy += 1
        else:
            n_watch += 1
        carried += 1 if row["carried"] else 0
        picks.append(row)

    return Slate(
        picks=picks, buy_count=n_buy, watch_count=n_watch,
        ranked_buys=list(recommendations), ranked_watch=list(watchlist),
        selection_note=_recorded_note(n_buy, n_watch, carried, payload),
        carried_count=carried,
    )


def _recorded_note(n_buy: int, n_watch: int, carried: int, payload) -> str:
    total = n_buy + n_watch
    if not total:
        return "No names qualified or came close enough to monitor."
    parts = []
    if n_buy:
        parts.append(f"{n_buy} qualifying {'setup' if n_buy == 1 else 'setups'}")
    if n_watch:
        parts.append(f"{n_watch} closest {'candidate' if n_watch == 1 else 'candidates'}")
    note = " and ".join(parts) + "."
    if carried:
        note += (f" {carried} of {total} {'was' if carried == 1 else 'were'} held "
                 f"from the previous run rather than picked again today.")
    departures = list(payload.get("slate_departures") or [])
    if departures:
        note += (f" {len(departures)} left the screen: "
                 + "; ".join(f"{d.get('ticker')} -- {d.get('reason')}"
                             for d in departures) + ".")
    return note


#: Engine vocabulary -> what the reader needs to do about it.
_EVENT_LABEL = {
    "index_removal": "No longer in the tradeable universe",
    "trading_suspension": "Not trading",
    "delisting": "Delisted",
}


def _position_alerts(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Open positions the run could not evaluate, in the reader's words.

    A name in this list is one the engine still considers held but produced no
    card for. It is the only place the screen can say so: the position is not
    among the picks, so without this it is invisible exactly when it matters.
    """
    alerts: List[Dict[str, Any]] = []
    for directive in payload.get("position_directives") or []:
        if not isinstance(directive, dict):
            continue
        event = str(directive.get("event") or "")
        if event == "none":
            continue
        alerts.append({
            "ticker": directive.get("ticker"),
            "label": _EVENT_LABEL.get(event, event.replace("_", " ").capitalize()),
            "exit_required": directive.get("action") == "force_exit",
            "detail": directive.get("reason") or "",
            "last_price": directive.get("last_tradeable_price"),
            "last_date": directive.get("last_tradeable_date"),
        })
    return alerts


def _concentration(picks: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Say when the whole shortlist is the same bet.

    Five names that all rank for the same reason are not five independent
    ideas -- they rise and fall together, and a reader holding all five is
    more concentrated than the count suggests. The model is momentum-heavy by
    construction, so this is common rather than exceptional, and it is exactly
    the sort of thing a list of five hides unless it is said.
    """
    if len(picks) < 3:
        return None
    leaders: Dict[str, int] = {}
    for pick in picks:
        top = next(
            (c["label"] for c in pick.get("evidence") or []
             if c.get("available") and c.get("verdict") in {"Strong", "Supportive"}),
            None,
        )
        if top:
            leaders[top] = leaders.get(top, 0) + 1
    if not leaders:
        return None
    label, count = max(leaders.items(), key=lambda kv: kv[1])
    if count < max(3, len(picks) - 1):
        return None
    return {
        "label": label,
        "count": count,
        "total": len(picks),
        "text": (
            f"{count} of {len(picks)} rank mainly on {label.lower()}. They are "
            f"one bet more than they look, and would tend to fall together."
        ),
    }


def _build_pick(
    card: Dict[str, Any],
    names: Dict[str, str],
    sectors: Dict[str, str],
    *,
    entry_rank: int,
    exit_rank: int,
) -> Dict[str, Any]:
    status = card.get("status") or (
        BUY if str(card.get("decision", "")).startswith("BUY") else WATCH
    )
    categories = build_evidence(card.get("factors") or {})
    agree, judged = confirmation_count(categories)
    ticker = str(card.get("ticker") or "")

    pick: Dict[str, Any] = {
        "position": card.get("slate_position"),
        "ticker": ticker,
        "company": _tidy_company(_company(card, names)),
        "sector": card.get("sector") if card.get("sector") not in (None, "Unknown")
                  else sectors.get(ticker),
        "status": status,
        "price": card.get("last_close"),
        # The engine's own band, renamed. `signal_strength_band` is not a
        # probability and must never be presented as one.
        "strength": _STRENGTH_LABEL.get(str(card.get("strength")), str(card.get("strength"))),
        "rank": card.get("model_rank"),
        "confirmation": {"agree": agree, "judged": judged},
        "highlights": category_summary(categories)[:4],
        # Whether this row is a new idea or a position the engine has been
        # carrying. The screen showed neither, so a name that had been held for
        # six sessions and one first seen this morning were indistinguishable,
        # and the "hold ~14 sessions" figure on the card had nothing to sit
        # against.
        "carried": bool(card.get("carried")),
        "shown_since": card.get("shown_since"),
        "why_shown": card.get("slate_reason") or "",
    }
    narrative = build_narrative(
        card, categories, status=status,
        entry_rank=entry_rank, exit_rank=exit_rank,
    )
    pick.update({
        "thesis": narrative["thesis"],
        "what_would_change": narrative["what_would_change"],
        "supporting": narrative["supporting"],
        "detracting": narrative["detracting"],
        "evidence": [
            {
                "key": c.key, "label": c.label, "verdict": c.verdict,
                "detail": c.detail, "available": c.available,
                "signals": [asdict(s) for s in c.signals],
            }
            for c in categories
        ],
        # THE TWO MODELS, SEPARATED. `themes` is the v3 composite -- what
        # ordered this book. `model_reading` is the fitted 26-factor model,
        # which is recorded and monitored and chose nothing. They arrived as
        # one flat dict and the panel rendered all eleven rows under a single
        # heading, counting their members together, which is where "12
        # measured factors" came from.
        "themes": _contributions(card.get("factors") or {}, tier="theme"),
        "model_reading": _contributions(card.get("factors") or {},
                                        tier="model_secondary"),
        # Kept for anything still reading the old key: the rows that DECIDE,
        # falling back to whatever tier exists when v3 is not the ranker.
        "contributions": (_contributions(card.get("factors") or {}, tier="theme")
                          or _contributions(card.get("factors") or {})),
        "earnings_note": card.get("earnings_note"),
        "entry_admissible": bool(card.get("entry_admissible", True)),
        "entry_block_reason": card.get("entry_block_reason"),
        # The card shows the largest few. Without the denominator a reader
        # cannot tell whether that is most of the model or a corner of it.
        "factors_considered": len(card.get("factors") or {}),
        # WHAT WAS COMPUTED BUT NOT PRICED. The engine ranks 26 columns and
        # fits at most seven themes over them, so "26 factors" and "5 themes"
        # are both true and neither alone is honest. The panel showed five and
        # said nothing about the other twenty-one.
        "unscored": _unscored_note(card),
        "levels": _levels(card),
        "holding_period": card.get("holding_period"),
        "cost_note": card.get("cost_note"),
        # Kept for the research drawer, not the card.
        "technical": {
            "score": card.get("score"),
            "percentile": card.get("percentile"),
            "display_rank": card.get("rank"),
            "why_raw": card.get("why"),
            "against_raw": card.get("against"),
            "cleared": card.get("cleared"),
            "not_testable": card.get("not_testable"),
            "research_basis": card.get("research_basis"),
            "warning": card.get("warning"),
        },
    })
    return pick


def _expectancy(cards: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The study's population figures, taken from the first card that has them.

    They are IDENTICAL on every card by construction -- they describe the
    trades the configuration took in the study, not this name -- which is
    exactly why they belong to the list rather than to a row.
    """
    for c in cards or []:
        tp = c.get("trade_plan") or {}
        if not tp:
            continue
        out = {
            "sample_trades": tp.get("sample_trades"),
            "sample_period": tp.get("sample_period"),
            "expected_return_pct": tp.get("expected_return_pct"),
            "median_return_pct": tp.get("median_return_pct"),
            "expected_excess_pct": tp.get("expected_excess_pct"),
            "median_excess_pct": tp.get("median_excess_pct"),
            "win_rate": tp.get("probability_of_profit"),
            "beat_rate": tp.get("probability_of_beating_benchmark"),
            "assumed_cost_bps": tp.get("assumed_cost_bps"),
            "typical_hold_sessions": tp.get("expected_hold_sessions"),
            "backstop_hold_sessions": tp.get("planned_hold_sessions"),
            "cadence_sessions": tp.get("cadence_sessions"),
        }
        return out if any(v is not None for v in out.values()) else None
    return None


def _unscored_note(card: Dict[str, Any]) -> Optional[str]:
    """The columns the run measured and deliberately did not price.

    Three different reasons, and collapsing them would be the misleading part:

      dropped for coverage  the fundamentals. The value and quality themes are
                            only built when their inputs span enough of the
                            panel; below that floor they are absent, not zero.
      unscored diagnostics  amihud and turnover_ratio. They are one factor
                            measured from two sides and the side they measure
                            is the illiquidity premium, which a manual
                            executor PAYS rather than collects. The universe
                            screen holds that as a floor instead.
      unscored control      log_mcap, carried so the size of what is being
                            ranked is visible without paying it a coefficient.
    """
    from ..features.crossmodel import FEATURE_COLUMNS
    from ..features.families import UNSCORED_CONTROLS, UNSCORED_DIAGNOSTICS

    # ONLY THE FITTED MODEL'S COLUMNS. This counted members across every entry
    # in `factors`, and since v3 that dict also holds the composite's own five
    # themes -- so the sum was the two models added together and the "columns
    # not priced" arithmetic below went negative.
    factors = {k: v for k, v in (card.get("factors") or {}).items()
               if (v or {}).get("tier") in (None, "model", "model_secondary")}
    members = sum(len((f or {}).get("members") or []) for f in factors.values())
    if not members:
        return None
    computed = len(FEATURE_COLUMNS)
    diagnostics = len(UNSCORED_DIAGNOSTICS)
    controls = len(UNSCORED_CONTROLS)
    absent = computed - members - diagnostics - controls
    bits = [f"The run ranks {computed} columns in all."]
    if absent > 0:
        bits.append(
            f"{absent} are fundamental and were dropped this run because they "
            f"do not span enough of the training panel -- absent, not zero.")
    bits.append(
        f"{diagnostics} ({', '.join(c[:-2] for c in UNSCORED_DIAGNOSTICS)}) "
        f"measure the illiquidity premium, which a manual executor pays rather "
        f"than collects, so the universe screen holds it as a floor instead of "
        f"the model paying it a coefficient.")
    bits.append(
        f"{controls} ({', '.join(c[:-2] for c in UNSCORED_CONTROLS)}) is "
        f"carried so the size of what is being ranked is visible, without "
        f"being priced.")
    return " ".join(bits)


#: The v3 themes, which are NOT the fitted model's themes even where the word
#: is the same. `reversal` in FAMILY_MAP is the estimator's one-month move;
#: v3's `reversal` is four factors oriented at ten sessions. Sharing a label
#: between them put two different measurements under one name in one panel.
#:
#: THE LABEL DESCRIBES THE FITTED SIGN, NOT THE THEME'S ACADEMIC NAMESAKE.
#: The internal theme KEY is frozen (`features/v3.py:THEMES`, hashed into
#: `config_sha256` and matched by `tests/test_v3_score.py`) and cannot change
#: without re-earning the sealed holdouts. The reader's LABEL can and must,
#: because two keys are fitted with signs that invert their namesake:
#:
#:   key "quality"   = (net_margin -1, margin_stability -1). A POSITIVE
#:                     contribution therefore means the name has LOW and
#:                     UNSTABLE margins. Displaying that as "Business quality"
#:                     tells a reader the opposite of what raised the score, so
#:                     it is labelled for what it prices: a low-margin tilt,
#:                     measured on the ~19% of names that carry fundamentals.
#:   key "ownership" = delivery-fraction factors only. Delivery % is the share
#:                     of traded volume taken to demat rather than squared off
#:                     intraday; it is not a holdings/ownership measurement and
#:                     is mechanically tied to turnover and volatility, so it is
#:                     labelled "Delivery strength", not "ownership".
#:
#: See the 2026-09 quant audit, section D. Changing a LABEL is a presentation
#: fix and does not touch the frozen model; changing a SIGN would.
V3_THEME_LABELS: Dict[str, str] = {
    "momentum":  "Momentum",
    "quality":   "Low-margin tilt",
    "ownership": "Delivery strength",
    "risk":      "Downside risk",
    "reversal":  "Short-horizon reversal",
}


def _contributions(factors: Dict[str, Any], top: Optional[int] = None,
                   tier: Optional[str] = None) -> List[Dict[str, Any]]:
    """What each theme actually added to the score, largest first.

    THE CONTRIBUTION IS READ, NOT RECOMPUTED. This multiplied `standardised x
    weight` and ignored the `contribution` the engine had already serialised
    -- which is right for the fitted model, where the coefficient is the
    weight, and WRONG for the v3 composite, whose weights renormalise over the
    themes a name actually has. Measured on a live card: every v3 theme read
    19% low, uniformly, because the name was scored on four themes of five and
    the client re-multiplied by the undiluted 40%. `rundetail` serialises the
    contribution precisely so the client does not have to do this arithmetic;
    it is now used.

    `tier` selects one model's rows: `v3_theme` for what ORDERS the book,
    `model_secondary` for the fitted 26-factor reading that only watches it.
    """
    from .evidence import FAMILY_MAP, FACTOR_MAP

    rows: List[Dict[str, Any]] = []
    for name, detail in factors.items():
        if not (detail or {}).get("available", False):
            continue
        if tier is not None and (detail or {}).get("tier") != tier:
            continue
        sd = detail.get("standardised")
        weight = detail.get("weight")
        if not isinstance(sd, (int, float)) or not isinstance(weight, (int, float)):
            continue
        nominal = detail.get("nominal_weight")
        served = detail.get("contribution")
        contribution = (round(float(served), 5)
                        if isinstance(served, (int, float))
                        else round(float(sd) * float(weight), 5))
        fam = FAMILY_MAP.get(name)
        mapped = FACTOR_MAP.get(name)
        if (detail or {}).get("tier") == "theme":
            fam = None
            mapped = (None, V3_THEME_LABELS.get(name, name.title()))
        rows.append({
            "factor": name,
            "tier": (detail or {}).get("tier"),
            # The reader's name for the theme. `lottery` is the literature's
            # word (Bali, Cakici & Whitelaw 2011) and it reads as a guess
            # rather than a measurement; what it measures is how lottery-like
            # the payoff is, and the engine prices it NEGATIVELY -- it prefers
            # the calm name.
            "label": fam[1] if fam else (mapped[1] if mapped else name),
            "category": fam[0] if fam else (mapped[0] if mapped else None),
            "z": round(float(sd), 3),
            "coefficient": round(float(weight), 5),
            # WHAT THE THEME WAS DECLARED AT, beside what it ran at. The v3
            # blend renormalises over the themes a name has, so `quality` is
            # declared 18.99% and applied at 0% to the ~79% of names with no
            # fundamentals, while momentum's declared 40% runs at about 48%.
            # Showing only the declared figure states a weight the model did
            # not use; showing only the effective one hides that it moved.
            "nominal_coefficient": (round(float(nominal), 5)
                                    if isinstance(nominal, (int, float))
                                    else None),
            "contribution": contribution,
            "raw": detail.get("raw"),
            # The measured factors this theme averages, so the panel can show
            # WHICH one moved instead of repeating the theme with more columns.
            "members": [
                {"name": m.get("name"), "rank": m.get("rank"),
                 "description": m.get("description")}
                for m in ((detail or {}).get("members") or [])
            ],
        })
    rows.sort(key=lambda r: -abs(r["contribution"]))
    return rows if top is None else rows[:top]


#: Which price level each disarmable exit rung owns. A level whose rung is not
#: armed is not a level this run will act on, so the screen must not show it as
#: one.
_LEVEL_RUNG = {
    "invalidation": "thesis_invalidation",
    "target_1": "target_achieved",
    "target_2": "target_achieved",
}


def _armed_rungs(card: Dict[str, Any]) -> set:
    """The exit rungs this run actually armed, read off the recorded list.

    The list is what Stage 7 emitted for THIS run, so it survives a payload
    written under a different configuration -- which is the case that matters,
    because a stale payload is exactly when the screen would otherwise present
    a retired exit as a live one.
    """
    armed = set()
    for line in (card.get("exits") or []):
        text = str(line)
        if text.startswith("NOT an exit"):
            continue
        head = text.split("--", 1)[0]
        head = head.split("(", 1)[0]
        rung = head.split(".", 1)[-1].strip() if "." in head else head.strip()
        if rung:
            armed.add(rung)
    return armed


def _levels(card: Dict[str, Any]) -> Dict[str, Any]:
    """Only levels the engine actually defines, and only those it will act on.

    Stage 7 still COMPUTES the target and the invalidation -- the ledger records
    them and `outcomes` scores historical trades against them, so they cannot
    simply stop existing. But `exit_hierarchy` disarms all three of
    thesis_invalidation, target_achieved and trailing_stop, so none of them can
    close a position any more. Shipping them in the view put a TARGET on the
    card as the headline upside and an INVALIDATION beside it labelled "thesis
    gone", both describing exits this engine measured, priced and then removed.

    A level is emitted only when its rung is armed. `stop` and `entry` have no
    rung to check: the stop is the disaster floor, which is always armed, and
    the entry zone is not an exit at all.
    """
    zone = card.get("entry_zone")
    armed = _armed_rungs(card)
    levels = {
        "entry": list(zone) if zone else None,
        "stop": card.get("stop"),
        "exits": card.get("exits") or [],
    }
    for name, rung in _LEVEL_RUNG.items():
        levels[name] = card.get(name) if rung in armed else None
    return levels


def _market(regime: Dict[str, Any]) -> Dict[str, Any]:
    """Three readings, not a dashboard."""
    compat = str(regime.get("compatibility") or "Unknown")
    headline = {
        "Favorable": "Constructive",
        "Neutral": "Mixed",
        "Unfavorable": "Defensive",
    }.get(compat, compat)
    # Two readings, both of which change the strategy's behaviour. Breadth and
    # the universe counts were removed: breadth restated the summary sentence
    # verbatim, and the funnel counts describe the run rather than the market.
    return {
        "headline": headline,
        "summary": _market_summary(regime, headline),
        "allows_new_positions": bool(regime.get("allow_new_entries", False)),
        "readings": [
            {"label": "Market trend", "value": regime.get("trend"),
             "detail": _trend_evidence(regime)},
            {"label": "Volatility", "value": regime.get("volatility"),
             "detail": _vol_evidence(regime)},
        ],
        "method": _METHOD_NOTE,
        "in_transition": bool(regime.get("transition", False)),
    }


#: What each headline word means for the strategy, not for the market. The
#: verdict comes from `RegimeCompatibility`, which is decided by three things:
#: whether new entries are allowed at all, the momentum multiplier, and
#: whether the regime is mid-transition. A word on its own is decoration --
#: this is what it is actually claiming.
_HEADLINE_MEANING = {
    "Constructive": (
        "Conditions of the kind this strategy has needed to work: trending, "
        "not in transition, and not suppressing momentum."
    ),
    "Mixed": (
        "Conditions are workable but weaker than the strategy prefers, so "
        "position sizes are reduced."
    ),
    "Defensive": (
        "Either the regime is turning or momentum is being penalised. The "
        "strategy holds back here."
    ),
}


#: Stated because the labels are conventional, not fitted. The 50/200-session
#: averages and the tercile split are standard practice -- a 200-day filter is
#: the usual long-term trend test and a VIX percentile is the usual volatility
#: read -- but the exact windows carry status UNVALIDATED in the config and
#: were never searched on this data.
_METHOD_NOTE = (
    "Trend requires two things to agree: the annualised slope of a 63-session "
    "log-price regression on NIFTY 200, and price above its 200-session "
    "average. Either alone gives a different wrong answer -- the average calls "
    "a flat market trending whenever price sits a hair above the line, and the "
    "slope calls every bounce inside a bear market an uptrend. Volatility is "
    "India VIX ranked against its own trailing 252 sessions, split at the "
    "33rd and 67th percentiles. These windows are conventional defaults, not "
    "values fitted on this data."
)


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _trend_evidence(regime: Dict[str, Any]) -> Optional[str]:
    """The two measurements the trend label is read off."""
    slope = regime.get("trend_slope_annualised")
    vs_slow = regime.get("index_vs_slow_ma_pct")
    bits: List[str] = []
    if isinstance(slope, (int, float)):
        bits.append(f"slope {slope * 100:+.1f}% annualised")
    if isinstance(vs_slow, (int, float)):
        bits.append(f"{vs_slow:+.1f}% vs its 200-session average")
    return " · ".join(bits) or None


def _vol_evidence(regime: Dict[str, Any]) -> Optional[str]:
    level = regime.get("vix_level")
    pct = regime.get("vix_percentile")
    bits: List[str] = []
    if isinstance(level, (int, float)):
        bits.append(f"India VIX {level:.1f}")
    if isinstance(pct, (int, float)):
        bits.append(f"{_ordinal(int(round(pct)))} percentile of the last 252 sessions")
    return " · ".join(bits) or None


def _market_summary(regime: Dict[str, Any], headline: str) -> str:
    """One sentence saying what the verdict means and what produced it.

    It deliberately does NOT restate the breadth percentage or the universe
    counts. Those were separate rows saying the same thing twice.
    """
    meaning = _HEADLINE_MEANING.get(headline, "")
    trend = str(regime.get("trend") or "").lower()
    vol = str(regime.get("volatility") or "").lower()
    facts: List[str] = []
    if trend:
        facts.append(trend)
    if vol:
        facts.append(f"volatility {vol}")
    tail = f" Today: {', '.join(facts)}." if facts else ""
    if regime.get("transition"):
        tail += " The regime is in transition, which is itself a caution."
    if not regime.get("allow_new_entries", True):
        tail += " New positions are not being opened."
    return (meaning + tail).strip() or "Market conditions could not be assessed."


def _journey(funnel: Dict[str, Any], slate) -> List[Dict[str, Any]]:
    """The narrowing, in the order it happened."""
    steps: List[Dict[str, Any]] = []
    previous: Optional[int] = None
    for key, label in FUNNEL_LABELS.items():
        if key not in funnel:
            continue
        value = funnel.get(key)
        if not isinstance(value, int):
            continue
        removed = (previous - value) if isinstance(previous, int) else None
        steps.append({
            "key": key, "label": label, "count": value,
            "detail": FUNNEL_DETAIL.get(key, ""),
            "removed": removed if removed and removed > 0 else None,
        })
        previous = value
    if steps:
        steps.append({
            "key": "slate",
            "label": "Shown on this screen",
            "count": len(slate.picks),
            "detail": slate.selection_note,
            "removed": None,
        })
    return steps


def _data_state(payload: Dict[str, Any], flags: Sequence[str]) -> Dict[str, Any]:
    return {
        "as_of": payload.get("as_of_date"),
        "generated_at": payload.get("generated_at"),
        "engine_version": payload.get("engine_version"),
        "config_version": payload.get("config_version"),
        "flags": list(flags),
        "complete": not flags,
    }
