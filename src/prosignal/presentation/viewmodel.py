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
        "journey": _journey(payload.get("funnel") or {}, slate),
        "data": _data_state(payload, flags),
        "disclaimer": payload.get("disclaimer"),
        "confidence_note": payload.get("probability_note"),
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
    breadth = regime.get("breadth_pct")
    return {
        "headline": headline,
        "summary": _market_summary(regime),
        "allows_new_positions": bool(regime.get("allow_new_entries", False)),
        "readings": [
            {"label": "Market trend", "value": regime.get("trend")},
            {"label": "Breadth",
             "value": regime.get("breadth_state"),
             "detail": (f"{breadth:.0f}% of names above their trend line"
                        if isinstance(breadth, (int, float)) else None)},
            {"label": "Volatility", "value": regime.get("volatility")},
        ],
        "in_transition": bool(regime.get("transition", False)),
    }


def _market_summary(regime: Dict[str, Any]) -> str:
    trend = str(regime.get("trend") or "").lower()
    breadth = regime.get("breadth_pct")
    bits: List[str] = []
    if trend:
        bits.append(f"The broader market is in an {trend}" if trend.startswith(("u", "o"))
                    else f"The broader market is in a {trend}")
    if isinstance(breadth, (int, float)):
        bits.append(f"with {breadth:.0f}% of names above their trend line")
    text = " ".join(bits).strip()
    if not text:
        return "Market conditions could not be assessed."
    if not regime.get("allow_new_entries", True):
        return text + ". Conditions do not currently support opening new positions."
    return text + "."


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
