"""Schema validation for feeds entering the curated store.

A column-presence check catches a renamed field. It does not catch the failure
that actually corrupts a store: a field that is still present, still named
correctly, and now carries something else. NSE has reordered and repurposed
columns across the legacy-to-UDiFF transition, and a frame whose ``close``
column silently holds turnover parses without complaint, writes without
complaint, and produces a backtest that is wrong in a way no test of the model
would find.

So each feed declares ranges as well as names, and the ranges are chosen to be
violated by misalignment rather than by unusual markets. A price column holding
volume fails ``0 < value < 1e7`` immediately; a price column holding an unusual
price does not.

Violations raise. A feed whose shape we cannot verify is not written to the
curated store under a guess about what its columns mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.errors import IntegrityError

__all__ = ["ColumnRule", "FeedSchema", "SCHEMAS", "validate_feed"]


@dataclass(frozen=True)
class ColumnRule:
    """One column: whether it must exist, and what values are credible."""

    name: str
    required: bool = True
    numeric: bool = False
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    #: Fraction of non-null values allowed to sit outside the range before the
    #: feed is rejected. Small but non-zero: a single bad print is a data point,
    #: a systematic breach is a format change.
    tolerance: float = 0.01


@dataclass(frozen=True)
class FeedSchema:
    """Expected shape of one feed, with cross-column invariants."""

    name: str
    columns: Sequence[ColumnRule]
    #: Invariants across columns, as (description, callable returning a mask of
    #: violating rows).
    invariants: Sequence[Tuple[str, object]] = field(default_factory=tuple)
    min_rows: int = 1


def _prices_ohlc_invariants():
    return (
        ("high < low", lambda d: d["high"] < d["low"]),
        ("close outside the high-low range",
         lambda d: (d["close"] > d["high"]) | (d["close"] < d["low"])),
        ("open outside the high-low range",
         lambda d: (d["open"] > d["high"]) | (d["open"] < d["low"])),
    )


#: Upper bounds are deliberately loose. They exist to catch a volume or a
#: turnover landing in a price column, not to judge a share price.
_PRICE_MAX = 5_000_000.0

SCHEMAS: Dict[str, FeedSchema] = {
    "prices": FeedSchema(
        name="prices",
        columns=(
            ColumnRule("date"),
            ColumnRule("symbol"),
            ColumnRule("series"),
            ColumnRule("open", numeric=True, minimum=0.0, maximum=_PRICE_MAX),
            ColumnRule("high", numeric=True, minimum=0.0, maximum=_PRICE_MAX),
            ColumnRule("low", numeric=True, minimum=0.0, maximum=_PRICE_MAX),
            ColumnRule("close", numeric=True, minimum=0.0, maximum=_PRICE_MAX),
            ColumnRule("volume", numeric=True, minimum=0.0),
            ColumnRule("turnover", numeric=True, minimum=0.0),
        ),
        invariants=_prices_ohlc_invariants(),
    ),
    "delivery": FeedSchema(
        name="delivery",
        columns=(
            ColumnRule("date"),
            ColumnRule("symbol"),
            # A delivered fraction outside 0-100 means the column is not a
            # percentage, which is the misalignment worth catching here.
            ColumnRule("deliv_pct", numeric=True, minimum=0.0, maximum=100.0),
            ColumnRule("deliv_qty", numeric=True, minimum=0.0),
        ),
    ),
    "indices": FeedSchema(
        name="indices",
        columns=(
            ColumnRule("date"),
            ColumnRule("index_name"),
            ColumnRule("close", numeric=True, minimum=0.0, maximum=1_000_000.0),
        ),
    ),
    "corporate_actions": FeedSchema(
        name="corporate_actions",
        columns=(
            ColumnRule("symbol"),
            ColumnRule("ex_date"),
            # A ratio of zero would zero out every pre-ex price.
            ColumnRule("ratio", numeric=True, minimum=1e-6, maximum=1000.0),
        ),
    ),
}


def validate_feed(
    frame: pd.DataFrame,
    schema: FeedSchema,
    context: str = "",
) -> None:
    """Raise IntegrityError unless ``frame`` matches ``schema``.

    Raising rather than dropping rows: a frame that fails these checks is not
    a frame with some bad rows, it is a frame we have misunderstood, and
    salvaging part of it would write the misunderstanding to disk.
    """
    where = f" ({context})" if context else ""
    if frame is None:
        raise IntegrityError(f"{schema.name}{where}: no frame to validate", feed=schema.name)
    if len(frame) < schema.min_rows:
        raise IntegrityError(
            f"{schema.name}{where}: {len(frame)} rows, at least {schema.min_rows} expected",
            feed=schema.name,
        )

    missing = [c.name for c in schema.columns if c.required and c.name not in frame.columns]
    if missing:
        raise IntegrityError(
            f"{schema.name}{where}: missing required columns {sorted(missing)}. "
            f"Present: {sorted(map(str, frame.columns))}",
            feed=schema.name,
        )

    problems: List[str] = []
    for rule in schema.columns:
        if rule.name not in frame.columns:
            continue
        series = frame[rule.name]
        if not rule.numeric:
            continue
        values = pd.to_numeric(series, errors="coerce")
        present = values.notna()
        if not present.any():
            problems.append(f"{rule.name}: no numeric values at all")
            continue
        if series.notna().any():
            unparsed = float((series.notna() & ~present).mean())
            if unparsed > rule.tolerance:
                problems.append(f"{rule.name}: {unparsed:.1%} of values are not numeric")
        outside = pd.Series(False, index=frame.index)
        if rule.minimum is not None:
            outside |= present & (values < rule.minimum)
        if rule.maximum is not None:
            outside |= present & (values > rule.maximum)
        breach = float(outside.sum()) / float(present.sum())
        if breach > rule.tolerance:
            bounds = f"[{rule.minimum}, {rule.maximum}]"
            problems.append(
                f"{rule.name}: {breach:.1%} of values outside {bounds} "
                f"(observed {values[present].min():.4g} to {values[present].max():.4g})"
            )

    have = {c.name for c in schema.columns if c.name in frame.columns}
    for description, predicate in schema.invariants:
        try:
            if not {"open", "high", "low", "close"}.issubset(have):
                continue
            violating = predicate(frame)
            rate = float(pd.Series(violating).fillna(False).mean())
        except Exception as exc:                        # a rule that cannot run
            problems.append(f"invariant {description!r} could not be evaluated: {exc}")
            continue
        if rate > 0.01:
            problems.append(f"invariant violated in {rate:.1%} of rows: {description}")

    if problems:
        raise IntegrityError(
            f"{schema.name}{where} failed schema validation:\n  - "
            + "\n  - ".join(problems),
            feed=schema.name,
        )
