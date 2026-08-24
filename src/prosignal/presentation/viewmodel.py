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

from .evidence import build_evidence, category_summary, confirmation_count
from .narrative import build_narrative
from .selection import BUY, SLOTS, WATCH, select_slate

#: Engine vocabulary -> the words the interface uses. The funnel keys are the
#: engine's own and would otherwise reach the screen verbatim.
FUNNEL_LABELS: Dict[str, str] = {
    "universe_considered": "Stocks evaluated",
    "passed_eligibility": "Passed data and liquidity checks",
    "scored": "Ranked by the model",
    "survived_defense": "Cleared the risk checks",
    "triggered": "Met the entry criteria",
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
    """
    from .evidence import FACTOR_MAP

    seen: set = set()
    for pick in picks:
        seen.update((pick.get("factors") or {}).keys())
    if not seen:
        return {"model": "unknown", "validated": False,
                "note": "No factor detail was recorded for this run."}
    known = seen & set(FACTOR_MAP)
    if known:
        return {"model": "cross-sectional", "validated": True, "note": None}
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
) -> Dict[str, Any]:
    """Build the screen's data from one completed analysis payload."""
    names = company_names or {}
    sector_map = sectors or {}

    recommendations = list(payload.get("recommendations") or [])
    watchlist = list(payload.get("watchlist") or [])

    flags = list(payload.get("data_quality_flags") or [])
    slate = select_slate(recommendations, watchlist, slots=slots)

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
        "scorer": _scorer_used(recommendations + watchlist),
        "concentration": _concentration(picks),
        "journey": _journey(payload.get("funnel") or {}, slate),
        "data": _data_state(payload, flags),
        "disclaimer": payload.get("disclaimer"),
        "confidence_note": payload.get("probability_note"),
    }


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
        "contributions": _contributions(card.get("factors") or {}),
        # The card shows the largest few. Without the denominator a reader
        # cannot tell whether that is most of the model or a corner of it.
        "factors_considered": len(card.get("factors") or {}),
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


def _contributions(factors: Dict[str, Any], top: int = 8) -> List[Dict[str, Any]]:
    """What each factor actually added to the score, largest first.

    contribution = standardised loading x fitted coefficient. This is the
    arithmetic behind the ranking and it was previously only available as
    prose. The factor's own name is kept alongside the readable one -- someone
    checking the model against its source needs the identifier, not a
    paraphrase of it.
    """
    from .evidence import FACTOR_MAP

    rows: List[Dict[str, Any]] = []
    for name, detail in factors.items():
        if not (detail or {}).get("available", False):
            continue
        sd = detail.get("standardised")
        weight = detail.get("weight")
        if not isinstance(sd, (int, float)) or not isinstance(weight, (int, float)):
            continue
        mapped = FACTOR_MAP.get(name)
        rows.append({
            "factor": name,
            "label": mapped[1] if mapped else name,
            "category": mapped[0] if mapped else None,
            "z": round(float(sd), 3),
            "coefficient": round(float(weight), 5),
            "contribution": round(float(sd) * float(weight), 5),
            "raw": detail.get("raw"),
        })
    rows.sort(key=lambda r: -abs(r["contribution"]))
    return rows[:top]


def _levels(card: Dict[str, Any]) -> Dict[str, Any]:
    """Only levels the engine actually defines. Nothing is derived here."""
    zone = card.get("entry_zone")
    return {
        "entry": list(zone) if zone else None,
        "stop": card.get("stop"),
        "invalidation": card.get("invalidation"),
        "target_1": card.get("target_1"),
        "target_2": card.get("target_2"),
        "exits": card.get("exits") or [],
    }


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
