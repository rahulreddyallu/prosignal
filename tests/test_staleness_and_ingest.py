"""The store must be measured against the calendar, not against itself.

Two defects, found together because the first hid the second.

`_manifest_from_store` built a TradingCalendar from the store and then measured
the store's own last row against it. age_sessions came back 0 for every feed on
every run, `stale_required()` was permanently empty, and the max_age_sessions
limits in parameters.yaml could never fire. `analyse run` performs no ingest, so
a store falling behind is the ordinary case rather than an exotic one -- the
engine would have issued confident signals from arbitrarily old prices with
every feed reporting green.

Underneath that, `data ingest` crashed on every invocation:
`_ingest_csv_feeds(self, as_of, calendar)` referenced `opts` and `symbols`,
neither of which it took. So the store could not be refreshed even deliberately.
"""

from __future__ import annotations

import ast
import datetime as dt
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# staleness
# =============================================================================


def test_weekdays_are_counted_against_today_not_against_the_store():
    from prosignal.pipeline import _sessions_behind

    friday = dt.date(2026, 8, 21)
    assert _sessions_behind(friday, dt.date(2026, 8, 22)) == 0, "Saturday adds none"
    assert _sessions_behind(friday, dt.date(2026, 8, 24)) == 1, "Monday is one"
    assert _sessions_behind(dt.date(2026, 8, 18), dt.date(2026, 8, 22)) == 3


def test_a_feed_ahead_of_today_is_not_negative_age():
    from prosignal.pipeline import _sessions_behind

    assert _sessions_behind(dt.date(2026, 9, 1), dt.date(2026, 8, 22)) == 0


def test_a_missing_last_date_is_not_treated_as_fresh():
    from prosignal.pipeline import _sessions_behind

    assert _sessions_behind(None, dt.date(2026, 8, 22)) == 0


def test_the_manifest_no_longer_measures_the_store_against_itself():
    """The regression. If age is computed from a store-derived calendar it is
    identically zero and the limits are decorative."""
    source = (ROOT / "src" / "prosignal" / "pipeline.py").read_text(encoding="utf-8")
    block = source[source.index("def _manifest_from_store"):]
    block = block[:block.index("def _frames")]
    assert "_sessions_behind(last)" in block, (
        "live staleness must be measured against today; measuring against a "
        "calendar built from the store returns 0 for every feed forever"
    )
    assert "live = as_of >=" in block, (
        "a deliberate historical run is legitimately behind today and must not "
        "be failed for it"
    )


# =============================================================================
# ingest
# =============================================================================


def _methods(path: Path, class_name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == class_name)
    return [n for n in cls.body if isinstance(n, ast.FunctionDef)]


def _bound_names(fn: ast.FunctionDef) -> set:
    names = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, (ast.For, ast.AugAssign, ast.AnnAssign)):
            target = getattr(node, "target", None)
            if isinstance(target, ast.Name):
                names.add(target.id)
        elif isinstance(node, ast.Try):
            names |= {h.name for h in node.handlers if h.name}
        elif isinstance(node, ast.With):
            names |= {i.optional_vars.id for i in node.items
                      if isinstance(i.optional_vars, ast.Name)}
        elif isinstance(node, (ast.ListComp, ast.GeneratorExp,
                               ast.DictComp, ast.SetComp)):
            names |= {g.target.id for g in node.generators
                      if isinstance(g.target, ast.Name)}
    return names


@pytest.mark.parametrize("name", ["opts", "calendar", "as_of", "universe", "symbols"])
def test_no_ingest_method_uses_a_name_it_never_receives(name):
    """The exact shape of the crash: a helper referencing a caller's local.

    `_ingest_csv_feeds` used `opts` and `symbols` while taking neither, so every
    `prosignal data ingest` raised NameError before touching the network. A
    signature change that drops a parameter still used in the body is silent
    until the line runs, which is why this is checked structurally.
    """
    path = ROOT / "src" / "prosignal" / "data" / "ingest.py"
    offenders = []
    for fn in _methods(path, "DataIngestor"):
        bound = _bound_names(fn)
        used = any(isinstance(n, ast.Name) and n.id == name
                   and isinstance(n.ctx, ast.Load) for n in ast.walk(fn))
        if used and name not in bound:
            offenders.append(f"{fn.name}:{fn.lineno}")
    assert not offenders, (
        f"these methods read {name!r} without taking or assigning it: "
        + ", ".join(offenders)
    )
