"""Write the case for a name from the evidence that actually produced it.

Every sentence here is assembled from values the engine computed for THIS
name. Nothing is drawn from a pool of plausible market prose. The test that
matters is the one asserting two different names do not produce the same
paragraph: a template with the ticker swapped would read as understanding and
convey nothing, and on a screen where the output is a position someone may
take, that is worse than saying less.

Where the engine has no basis for a claim, the claim is absent. The watchlist
sentence explaining what would move a name to BUY is derived from the actual
admission rule -- a rank band with hysteresis -- and not from a guess about
what the market might do next.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .evidence import Category, confirmation_count

#: Verdicts that count as the category backing the name.
_POSITIVE = {"Strong", "Supportive", "Contained"}
_NEGATIVE = {"Weak", "Soft", "Elevated"}


def _join(items: Sequence[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _lead(categories: Sequence[Category]) -> str:
    """Open on the strongest thing that is actually true of this name."""
    judged = [c for c in categories if c.available]
    if not judged:
        return "Too little of this name's data was usable to describe the setup."
    ranked = sorted(judged, key=lambda c: -_magnitude(c))
    best = ranked[0]
    if best.verdict in _POSITIVE and _magnitude(best) >= 0.35:
        return best.detail
    worst = ranked[0]
    if worst.verdict in _NEGATIVE:
        return worst.detail
    return "This name sits close to the market average across the areas measured."


def _magnitude(cat: Category) -> float:
    if not cat.signals:
        return 0.0
    total_w = sum(s.weight for s in cat.signals)
    if total_w > 0:
        return abs(sum(s.oriented_sd * s.weight for s in cat.signals) / total_w)
    return abs(sum(s.oriented_sd for s in cat.signals) / len(cat.signals))


def build_narrative(
    card: Dict[str, Any],
    categories: Sequence[Category],
    *,
    status: str,
    entry_rank: int,
    exit_rank: int,
) -> Dict[str, Any]:
    """The prose block for one name."""
    supporting = [c.label.lower() for c in categories
                  if c.available and c.verdict in _POSITIVE]
    detracting = [c.label.lower() for c in categories
                  if c.available and c.verdict in _NEGATIVE]
    missing = [c.label.lower() for c in categories if not c.available]
    agree, judged = confirmation_count(categories)
    rank = card.get("model_rank")

    sentences: List[str] = [_lead(categories)]

    if supporting:
        sentences.append(
            f"The ranking is supported by {_join(supporting)}."
            if len(supporting) > 1 else
            f"The ranking is supported by {supporting[0]}."
        )
    if detracting:
        sentences.append(
            f"Working against it: {_join(detracting)}."
        )
    if isinstance(rank, int):
        sentences.append(
            f"Across the names the model could rank today it places "
            f"{_ordinal(rank)}, "
            f"{'inside' if status == 'BUY' else 'outside'} the top "
            f"{entry_rank} the strategy opens positions in."
        )

    # The penalties are the most decision-relevant thing the engine produced
    # and the old screen folded them into a number that changed row order.
    penalties = _penalties(card)
    if penalties:
        sentences.append(
            f"One check argued against the setup: {penalties[0]}"
            if len(penalties) == 1 else
            f"{len(penalties)} checks argued against the setup."
        )

    if missing:
        sentences.append(
            f"{_join([m.capitalize() for m in missing])} could not be assessed, "
            f"so this rests on partial evidence."
        )

    return {
        "thesis": " ".join(s for s in sentences if s),
        "confirmation": {"agree": agree, "judged": judged},
        "supporting": supporting,
        "detracting": detracting,
        "unavailable": missing,
        "what_would_change": _what_would_change(
            card, categories, status=status,
            entry_rank=entry_rank, exit_rank=exit_rank,
        ),
    }


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _penalties(card: Dict[str, Any]) -> List[str]:
    """The Stage 5 checks that fired, in readable form.

    The engine stores these as `low_volume_breakout: latest session volume is
    0.33x its 20-session average, below the 1.50x participation this check
    requires (-0.10)`. The rule name and the coefficient are engine vocabulary;
    the measurement is the part a reader needs.
    """
    out: List[str] = []
    for entry in card.get("against") or []:
        text = str(entry)
        if text.startswith("No disconfirming evidence"):
            continue
        body = text.split(":", 1)[1].strip() if ":" in text else text
        # Drop the trailing score delta -- it is the number that used to
        # reorder the table, and it means nothing without the whole scale.
        if body.endswith(")") and "(-" in body:
            body = body[:body.rfind("(-")].strip()
        out.append(body.rstrip(".") + ".")
    return out


def _what_would_change(
    card: Dict[str, Any],
    categories: Sequence[Category],
    *,
    status: str,
    entry_rank: int,
    exit_rank: int,
) -> Optional[str]:
    """What would move this name across the line, from the actual rule.

    Admission is a rank band with hysteresis: a name enters at `entry_rank` and
    is only dropped once it leaves `exit_rank`. That is the real mechanism, so
    it is what the sentence describes. Anything about price levels or future
    volume would be invention.
    """
    rank = card.get("model_rank")
    if not isinstance(rank, int):
        return None

    if status == "BUY":
        parts = [
            f"The position is held while this name stays inside the top "
            f"{exit_rank}. It would be closed if it falls below that."
        ]
        weakest = _weakest(categories)
        if weakest:
            parts.append(
                f"The most likely route there is {weakest.label.lower()}, "
                f"which is already the softest part of the case."
            )
        exits = card.get("exits") or []
        level = _first_level(exits)
        if level:
            parts.append(f"The stated invalidation level is {level}.")
        return " ".join(parts)

    gap = rank - entry_rank
    if gap <= 0:
        # Ranked inside the band but not admitted: something other than rank
        # held it back, and the penalties are the visible reason.
        pens = _penalties(card)
        if pens:
            return (
                "It ranks inside the entry band but a check argued against it: "
                + pens[0]
                + " Clearing that would put it in contention."
            )
        return "It ranks inside the entry band and is close to qualifying."

    strongest_gap = _weakest(categories)
    tail = ""
    if strongest_gap:
        tail = (
            f" The softest part of its case is {strongest_gap.label.lower()}, "
            f"so that is where the ground would have to be made up."
        )
    return (
        f"It would need to climb {gap} {'place' if gap == 1 else 'places'} to "
        f"reach the top {entry_rank} the strategy opens positions in.{tail}"
    )


def _weakest(categories: Sequence[Category]) -> Optional[Category]:
    judged = [c for c in categories if c.available]
    if not judged:
        return None
    worst = min(judged, key=lambda c: _signed(c))
    return worst if _signed(worst) < 0.35 else None


def _signed(cat: Category) -> float:
    if not cat.signals:
        return 0.0
    total_w = sum(s.weight for s in cat.signals)
    if total_w > 0:
        return sum(s.oriented_sd * s.weight for s in cat.signals) / total_w
    return sum(s.oriented_sd for s in cat.signals) / len(cat.signals)


def _first_level(exits: Sequence[str]) -> Optional[str]:
    for entry in exits:
        text = str(entry)
        if "Rs" in text:
            start = text.find("(Rs")
            end = text.find(")", start)
            if start != -1 and end != -1:
                return text[start + 1:end]
    return None
