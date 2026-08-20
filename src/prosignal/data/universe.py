"""Point-in-time universe resolution.

Projecting today's NIFTY 200 list backwards removes every name that fell out of
the index after collapsing, which flatters a backtest for reasons unrelated to
the strategy. NSE publishes only the current list, so this module surfaces the
problem rather than hiding it.

Resolution order, best available first:

1. ``config/reference/index_membership.csv`` -- hand-maintained effective-dated
   membership transcribed from NSE reconstitution circulars. Genuinely
   point-in-time, preferred whenever it covers the requested date.
2. A dated snapshot taken on or before ``as_of``. The engine snapshots the live
   list on every run, so point-in-time membership accumulates going forward.
3. The most recent snapshot later than ``as_of``. Survivorship-biased by
   construction: sets ``survivorship_risk=True``, and under
   ``universe.pre_snapshot_policy: halt`` refuses to run, which is the correct
   setting for backtesting.

Listing dates from ``EQUITY_L.csv`` are applied on top, so a company cannot
appear before it was listed whatever a snapshot says.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import pandas as pd

from ..core.errors import IntegrityError
from ..core.logging import get_logger
from .store import DataStore
from .types import SYMBOL, normalise_symbol

__all__ = ["UniverseSnapshot", "UniverseResolver"]

log = get_logger(__name__)


@dataclass
class UniverseSnapshot:
    """The universe as it stood (or as best we can reconstruct it) on a date."""

    index_name: str
    as_of: dt.date
    symbols: List[str]
    sector_map: Dict[str, str] = field(default_factory=dict)
    company_names: Dict[str, str] = field(default_factory=dict)
    isin_map: Dict[str, str] = field(default_factory=dict)
    source: str = "unknown"
    survivorship_risk: bool = False
    note: Optional[str] = None
    excluded_not_yet_listed: List[str] = field(default_factory=list)
    excluded_manual: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.symbols)

    def sector_of(self, symbol: str) -> str:
        return self.sector_map.get(normalise_symbol(symbol), "Unknown")

    def to_dict(self) -> Dict[str, object]:
        return {
            "index_name": self.index_name,
            "as_of": self.as_of.isoformat(),
            "size": len(self.symbols),
            "source": self.source,
            "survivorship_risk": self.survivorship_risk,
            "note": self.note,
            "excluded_not_yet_listed": len(self.excluded_not_yet_listed),
            "excluded_manual": len(self.excluded_manual),
        }


class UniverseResolver:
    """Resolves index membership for a decision date, honestly."""

    def __init__(self, store: DataStore, config: "object") -> None:
        self.store = store
        self.cfg = config

    # =====================================================================
    # public
    # =====================================================================
    def resolve(
        self,
        index_name: str,
        as_of: dt.date,
        membership_csv: Optional[pd.DataFrame] = None,
        manual_exclusions: Optional[Sequence[str]] = None,
        pre_snapshot_policy: str = "flag",
    ) -> UniverseSnapshot:
        snap = self._resolve_membership(index_name, as_of, membership_csv)

        if snap.survivorship_risk and pre_snapshot_policy == "halt":
            raise IntegrityError(
                f"No point-in-time membership for {index_name} on {as_of}. The "
                f"only available snapshot is dated later, which makes the "
                f"universe survivorship-biased. universe.pre_snapshot_policy is "
                f"set to 'halt', so this run is refused. Either transcribe the "
                f"relevant NSE reconstitution circulars into "
                f"config/reference/index_membership.csv, or switch the policy to "
                f"'flag' if this is a forward/live run where today's list IS the "
                f"point-in-time list.",
                index_name=index_name,
                as_of=as_of.isoformat(),
                available_snapshots=[
                    d.isoformat() for d in self.store.universe_snapshot_dates(index_name)
                ][-5:],
            )

        self._apply_listing_dates(snap, as_of)
        self._apply_manual_exclusions(snap, manual_exclusions or [])
        return snap

    def resolve_liquidity_pit(
        self,
        as_of: dt.date,
        min_adtv_inr: float,
        lookback_sessions: int,
        max_names: int,
        min_history_sessions: int,
        min_price_inr: float,
        manual_exclusions: Optional[Sequence[str]] = None,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> UniverseSnapshot:
        """The universe as a trailing-liquidity screen, with no membership list.

        Every input is drawn from sessions at or before ``as_of``, so the set is
        point-in-time by construction. A name that collapsed is present for as
        long as it was liquid and disappears afterwards, which is what a live
        book would have experienced. Nothing here consults an index.
        """
        sessions = [d for d in self.store.price_sessions() if d <= as_of]
        if not sessions:
            raise IntegrityError(
                "no price sessions at or before the decision date; run "
                "`prosignal data ingest` before resolving a universe.",
                as_of=as_of.isoformat(),
            )
        window = sessions[-int(lookback_sessions):]
        px = self.store.read_prices(
            start=window[0], end=as_of, columns=["date", "symbol", "close", "turnover"]
        )
        if px.empty:
            raise IntegrityError(
                "no price rows in the liquidity window", as_of=as_of.isoformat()
            )
        px["date"] = pd.to_datetime(px["date"]).dt.normalize()

        # Median, not mean: one block deal should not buy a name a seat.
        adtv = px.groupby("symbol", observed=True)["turnover"].median()
        last = px.sort_values("date").groupby("symbol", observed=True)["close"].last()
        # History is measured from the listing date against the session list,
        # which is exact and costs one small read. Counting rows per symbol
        # across every year would rescan the whole price store on every run.
        listed_before = self._listed_at_least(sessions, int(min_history_sessions))

        eligible = adtv[adtv >= float(min_adtv_inr)].index
        keep = [
            s for s in eligible
            if float(last.get(s, 0.0)) >= float(min_price_inr)
            and s in listed_before
        ]
        ranked = adtv.reindex(keep).sort_values(ascending=False)
        symbols = [str(s) for s in ranked.head(int(max_names)).index]
        if not symbols:
            raise IntegrityError(
                f"the liquidity screen admitted no symbols on {as_of}: "
                f"turnover floor Rs {float(min_adtv_inr):,.0f}, price floor "
                f"Rs {float(min_price_inr):,.2f}, "
                f"{int(min_history_sessions)} sessions of history required.",
                as_of=as_of.isoformat(),
            )

        sectors = {s: (sector_map or {}).get(s, "Unknown") for s in symbols}
        known = sum(1 for v in sectors.values() if v != "Unknown")
        snap = UniverseSnapshot(
            index_name="LIQUIDITY-PIT",
            as_of=as_of,
            symbols=symbols,
            sector_map=sectors,
            source=f"liquidity_pit:adtv>={float(min_adtv_inr):.0f}",
            survivorship_risk=False,
            note=(
                f"point-in-time liquidity screen over {len(window)} sessions ending "
                f"{as_of}; {len(symbols)} names; sector known for {known} "
                f"({100.0 * known / len(symbols):.0f}%)"
            ),
        )
        self._apply_listing_dates(snap, as_of)
        self._apply_manual_exclusions(snap, manual_exclusions or [])
        log.info(
            "liquidity universe resolved",
            extra={"as_of": as_of.isoformat(), "size": len(snap.symbols),
                   "sector_coverage": round(100.0 * known / max(len(symbols), 1), 1)},
        )
        return snap

    def _listed_at_least(self, sessions: Sequence[dt.date], min_sessions: int) -> Set[str]:
        """Symbols listed early enough to have ``min_sessions`` of history.

        A symbol absent from the master is kept: the master is a convenience
        file, and dropping names because a reference feed is thin would silently
        shrink the universe for a reason unrelated to liquidity.
        """
        if len(sessions) <= min_sessions:
            cutoff = sessions[0]
        else:
            cutoff = sessions[-(min_sessions + 1)]
        master = self.store.read_equity_master()
        if master.empty or "listing_date" not in master.columns:
            return set()
        listing = (
            master.dropna(subset=["listing_date"])
            .assign(**{SYMBOL: lambda d: d[SYMBOL].map(normalise_symbol)})
            .set_index(SYMBOL)["listing_date"]
        )
        listing = listing[~listing.index.duplicated(keep="first")]
        early = set(listing[pd.to_datetime(listing) <= pd.Timestamp(cutoff)].index)
        return early | (set() if master.empty else set())

    def snapshot_current(
        self, index_name: str, as_of: dt.date, constituents: pd.DataFrame
    ) -> None:
        """Persist today's live constituent list as a dated snapshot."""
        if constituents is None or constituents.empty:
            return
        self.store.write_universe_snapshot(index_name, as_of, constituents)
        log.info(
            "universe snapshot written",
            extra={"index": index_name, "as_of": as_of.isoformat(), "size": len(constituents)},
        )

    # =====================================================================
    # resolution strategies
    # =====================================================================
    def _resolve_membership(
        self,
        index_name: str,
        as_of: dt.date,
        membership_csv: Optional[pd.DataFrame],
    ) -> UniverseSnapshot:
        # -- 1. hand-maintained effective-dated membership -------------------
        if membership_csv is not None and not membership_csv.empty:
            snap = self._from_membership_csv(index_name, as_of, membership_csv)
            if snap is not None:
                return snap

        # -- 2/3. dated snapshots -------------------------------------------
        available = self.store.universe_snapshot_dates(index_name)
        if not available:
            raise IntegrityError(
                f"No universe data for {index_name}. Run `prosignal data ingest` "
                f"first -- the engine will not invent a constituent list.",
                index_name=index_name,
            )

        on_or_before = [d for d in available if d <= as_of]
        if on_or_before:
            chosen = on_or_before[-1]
            frame = self.store.read_universe_snapshot(index_name, chosen)
            lag = (as_of - chosen).days
            note = (
                f"membership from snapshot dated {chosen} "
                f"({lag} calendar day(s) before the decision date)"
            )
            return self._frame_to_snapshot(
                index_name, as_of, frame, source=f"snapshot:{chosen}", note=note
            )

        chosen = available[0]
        frame = self.store.read_universe_snapshot(index_name, chosen)
        note = (
            f"SURVIVORSHIP RISK: earliest available snapshot is {chosen}, which is "
            f"AFTER the decision date {as_of}. Constituents that left the index "
            f"between those dates are invisible to this run, and names that "
            f"joined later are wrongly present. Treat any backtest result built "
            f"on this as unusable."
        )
        log.warning("survivorship risk", extra={"index": index_name, "as_of": as_of.isoformat()})
        snap = self._frame_to_snapshot(
            index_name, as_of, frame, source=f"snapshot:{chosen}", note=note
        )
        snap.survivorship_risk = True
        return snap

    def _from_membership_csv(
        self, index_name: str, as_of: dt.date, membership: pd.DataFrame
    ) -> Optional[UniverseSnapshot]:
        wanted = index_name.strip().upper()
        rows = membership[membership["index_name"].str.upper() == wanted]
        if rows.empty:
            return None

        ts = pd.Timestamp(as_of)
        # Coerce the date columns here rather than trusting the caller. A frame
        # read straight from CSV carries strings, and comparing a string to a
        # Timestamp raises -- which would break the one mechanism that can fix
        # survivorship bias, at the moment someone finally populated the file.
        rows = rows.copy()
        rows["effective_from"] = pd.to_datetime(rows["effective_from"], errors="coerce")
        rows["effective_to"] = pd.to_datetime(
            rows["effective_to"].replace("", pd.NA), errors="coerce"
        )
        rows = rows.dropna(subset=["effective_from"])
        if rows.empty:
            return None

        # Does the file actually cover this date? If its earliest effective_from
        # is after as_of, it does not, and we must fall through rather than
        # return a confidently wrong (empty) universe.
        if rows["effective_from"].min() > ts:
            return None

        active = rows[rows["effective_from"] <= ts]
        active = active[active["effective_to"].isna() | (active["effective_to"] > ts)]
        if active.empty:
            return None

        symbols = sorted(set(active[SYMBOL].map(normalise_symbol)))
        snap = UniverseSnapshot(
            index_name=index_name,
            as_of=as_of,
            symbols=symbols,
            source="index_membership.csv",
            note=(
                "point-in-time membership from your hand-maintained "
                "config/reference/index_membership.csv"
            ),
        )
        self._attach_sectors_from_latest_snapshot(snap)
        return snap

    def _frame_to_snapshot(
        self,
        index_name: str,
        as_of: dt.date,
        frame: Optional[pd.DataFrame],
        source: str,
        note: Optional[str],
    ) -> UniverseSnapshot:
        if frame is None or frame.empty:
            raise IntegrityError(
                f"universe snapshot for {index_name} is empty", source=source
            )
        symbols = sorted(set(frame[SYMBOL].map(normalise_symbol)))
        sector_map: Dict[str, str] = {}
        company_names: Dict[str, str] = {}
        isin_map: Dict[str, str] = {}
        for _, row in frame.iterrows():
            sym = normalise_symbol(row[SYMBOL])
            if "sector" in frame.columns and pd.notna(row.get("sector")):
                sector_map[sym] = str(row["sector"]).strip() or "Unknown"
            if "company_name" in frame.columns and pd.notna(row.get("company_name")):
                company_names[sym] = str(row["company_name"]).strip()
            if "isin" in frame.columns and pd.notna(row.get("isin")):
                isin_map[sym] = str(row["isin"]).strip()
        return UniverseSnapshot(
            index_name=index_name,
            as_of=as_of,
            symbols=symbols,
            sector_map=sector_map,
            company_names=company_names,
            isin_map=isin_map,
            source=source,
            note=note,
        )

    def _attach_sectors_from_latest_snapshot(self, snap: UniverseSnapshot) -> None:
        """Sector labels come from the newest snapshot we hold.

        Stated plainly because it matters: NSE's constituent file carries only
        the CURRENT industry label, so a company reclassified since your data
        window inherits its new sector for old dates. That is a known,
        acknowledged deviation from the research program's "historical sector
        classification" requirement, and the only clean fix is a paid
        point-in-time classification feed dropped into the CSV importer.
        """
        dates = self.store.universe_snapshot_dates(snap.index_name)
        if not dates:
            return
        frame = self.store.read_universe_snapshot(snap.index_name, dates[-1])
        if frame is None or frame.empty:
            return
        for _, row in frame.iterrows():
            sym = normalise_symbol(row[SYMBOL])
            if sym not in snap.symbols:
                continue
            if "sector" in frame.columns and pd.notna(row.get("sector")):
                snap.sector_map.setdefault(sym, str(row["sector"]).strip() or "Unknown")
            if "company_name" in frame.columns and pd.notna(row.get("company_name")):
                snap.company_names.setdefault(sym, str(row["company_name"]).strip())
        snap.note = (snap.note or "") + (
            " | sector labels are current-vintage, not historical"
        )

    # =====================================================================
    # filters applied on top of membership
    # =====================================================================
    def _apply_listing_dates(self, snap: UniverseSnapshot, as_of: dt.date) -> None:
        master = self.store.read_equity_master()
        if master.empty or "listing_date" not in master.columns:
            return
        listing = (
            master.dropna(subset=["listing_date"])
            .assign(**{SYMBOL: lambda d: d[SYMBOL].map(normalise_symbol)})
            .set_index(SYMBOL)["listing_date"]
        )
        listing = listing[~listing.index.duplicated(keep="first")]
        ts = pd.Timestamp(as_of)
        not_yet: List[str] = []
        for sym in list(snap.symbols):
            listed_on = listing.get(sym)
            if listed_on is not None and pd.notna(listed_on) and pd.Timestamp(listed_on) > ts:
                not_yet.append(sym)
        if not_yet:
            keep: Set[str] = set(snap.symbols) - set(not_yet)
            snap.symbols = sorted(keep)
            snap.excluded_not_yet_listed = sorted(not_yet)
            log.info(
                "excluded names not yet listed at decision date",
                extra={"count": len(not_yet), "as_of": as_of.isoformat()},
            )

    def _apply_manual_exclusions(
        self, snap: UniverseSnapshot, exclusions: Sequence[str]
    ) -> None:
        if not exclusions:
            return
        blocked = {normalise_symbol(s) for s in exclusions}
        removed = sorted(blocked & set(snap.symbols))
        if removed:
            snap.symbols = sorted(set(snap.symbols) - blocked)
            snap.excluded_manual = removed
