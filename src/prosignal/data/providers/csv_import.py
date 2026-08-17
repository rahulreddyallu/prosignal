"""User-supplied CSV drop-in provider.

This is the honest answer to the feeds that no free source delivers reliably
for India -- promoter pledging above all, plus point-in-time fundamentals with
filing dates, confirmed earnings dates, and your own regulatory-event log.

The contract is deliberately blunt: **if the file is absent, the dependent
check reports NOT_TESTABLE.** It never quietly passes. An engine that treats
"I could not check pledging" as "pledging is fine" is worse than one that has
no pledging check at all, because it launders an unknown into a reassurance.

Every loader here returns a frame with the exact columns the engine expects,
empty if the file is missing, and raises only when a file exists but is
structurally wrong -- because a malformed file the user believes is working is
the dangerous case.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ...core.errors import DataError
from ...core.logging import get_logger
from ..types import CORPORATE_ACTION_COLUMNS, SYMBOL, normalise_symbol

__all__ = ["CsvImportProvider", "REFERENCE_TEMPLATES"]

log = get_logger(__name__)


#: Column contract for each optional reference file, used both for validation
#: and for writing the blank templates that ship with the project.
REFERENCE_TEMPLATES: Dict[str, List[str]] = {
    "pledging": [
        "symbol",
        "as_of_date",
        "pledged_pct_of_promoter_holding",
        "promoter_holding_pct",
        "source",
    ],
    "fundamentals": [
        "symbol",
        "filing_date",
        "period_end",
        "revenue",
        "net_profit",
        "total_equity",
        "total_assets",
        "total_debt",
        "ebit",
        "interest_expense",
        "operating_cash_flow",
        "shares_outstanding",
    ],
    "earnings_calendar": ["symbol", "earnings_date", "confirmed", "source"],
    "corporate_actions": [
        "symbol",
        "ex_date",
        "action_type",
        "ratio_from",
        "ratio_to",
        "details",
    ],
    "regulatory_events": ["symbol", "event_date", "cooldown_days", "reason"],
    "index_membership": ["index_name", "symbol", "effective_from", "effective_to"],
}


def _read_csv(path: Path, required: List[str], label: str) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DataError(
            f"{label} file exists but could not be read: {path} ({exc}). "
            f"A malformed reference file is more dangerous than a missing one, "
            f"so this is fatal rather than skipped."
        ) from exc
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(
            f"{label} file {path} is missing required column(s) {missing}. "
            f"Expected header: {','.join(required)}"
        )
    if df.empty:
        return None
    return df


class CsvImportProvider:
    """Loads the optional reference CSVs named in ``providers.csv_import``."""

    name = "csv_import"

    def __init__(self, cfg: "object", project_root: Path) -> None:
        self.cfg = cfg
        self.root = Path(project_root)
        #: Feeds that were requested but had no file. Surfaced in the manifest.
        self.absent: List[str] = []

    def _path(self, attr: str) -> Path:
        rel = getattr(self.cfg, attr)
        p = Path(rel).expanduser()
        return p if p.is_absolute() else (self.root / p)

    def _note_absent(self, label: str, path: Path) -> None:
        self.absent.append(label)
        log.info(
            "optional reference file not present",
            extra={"feed": label, "expected_path": str(path)},
        )

    # =====================================================================
    # pledging  (Kalia 2024 -- crash-risk filter, never an alpha signal)
    # =====================================================================
    def load_pledging(self) -> pd.DataFrame:
        path = self._path("pledging_file")
        df = _read_csv(path, REFERENCE_TEMPLATES["pledging"], "pledging")
        if df is None:
            self._note_absent("pledging", path)
            return pd.DataFrame(columns=REFERENCE_TEMPLATES["pledging"])
        out = pd.DataFrame(
            {
                SYMBOL: df["symbol"].map(normalise_symbol),
                # SEBI-mandated DISCLOSURE date, not the reporting period end.
                "as_of_date": pd.to_datetime(df["as_of_date"], errors="coerce").dt.normalize(),
                "pledged_pct_of_promoter_holding": pd.to_numeric(
                    df["pledged_pct_of_promoter_holding"], errors="coerce"
                ),
                "promoter_holding_pct": pd.to_numeric(
                    df["promoter_holding_pct"], errors="coerce"
                ),
                "source": df.get("source", self.name),
            }
        )
        return out.dropna(subset=[SYMBOL, "as_of_date"]).reset_index(drop=True)

    # =====================================================================
    # fundamentals  (timestamped to FILING date -- the India leakage risk)
    # =====================================================================
    def load_fundamentals(self) -> pd.DataFrame:
        path = self._path("fundamentals_file")
        df = _read_csv(path, ["symbol", "filing_date"], "fundamentals")
        if df is None:
            self._note_absent("fundamentals", path)
            return pd.DataFrame(columns=REFERENCE_TEMPLATES["fundamentals"])

        out = pd.DataFrame({SYMBOL: df["symbol"].map(normalise_symbol)})
        out["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce").dt.normalize()
        out["period_end"] = pd.to_datetime(
            df.get("period_end", pd.Series(index=df.index)), errors="coerce"
        ).dt.normalize()
        for col in REFERENCE_TEMPLATES["fundamentals"][3:]:
            out[col] = pd.to_numeric(df.get(col, pd.Series(index=df.index)), errors="coerce")

        out = out.dropna(subset=[SYMBOL, "filing_date"])

        # Indian companies report with a lag. A filing date at or before its own
        # period end is impossible and is the classic look-ahead footgun.
        both = out.dropna(subset=["period_end"])
        bad = both[both["filing_date"] <= both["period_end"]]
        if not bad.empty:
            raise DataError(
                f"{len(bad)} fundamentals row(s) have filing_date <= period_end "
                f"(e.g. {bad.iloc[0][SYMBOL]}). Indian companies report with a "
                f"lag, so this means the file is timestamped to the fiscal "
                f"period rather than to public availability -- exactly the "
                f"look-ahead bias the point-in-time audit exists to prevent.",
                offending_symbols=bad[SYMBOL].head(10).tolist(),
            )
        return out.reset_index(drop=True)

    # =====================================================================
    # earnings calendar
    # =====================================================================
    def load_earnings_calendar(self) -> pd.DataFrame:
        path = self._path("earnings_calendar_file")
        df = _read_csv(path, ["symbol", "earnings_date"], "earnings_calendar")
        if df is None:
            self._note_absent("earnings_calendar", path)
            return pd.DataFrame(columns=REFERENCE_TEMPLATES["earnings_calendar"])
        out = pd.DataFrame(
            {
                SYMBOL: df["symbol"].map(normalise_symbol),
                "earnings_date": pd.to_datetime(
                    df["earnings_date"], errors="coerce"
                ).dt.normalize(),
                "confirmed": df.get("confirmed", True),
                "source": df.get("source", self.name),
            }
        )
        out["confirmed"] = out["confirmed"].astype(str).str.lower().isin(
            ["true", "1", "yes", "y"]
        )
        return out.dropna(subset=[SYMBOL, "earnings_date"]).reset_index(drop=True)

    # =====================================================================
    # corporate actions
    # =====================================================================
    def load_corporate_actions(self) -> pd.DataFrame:
        path = self._path("corporate_actions_file")
        df = _read_csv(path, ["symbol", "ex_date", "action_type"], "corporate_actions")
        if df is None:
            self._note_absent("corporate_actions", path)
            return pd.DataFrame(columns=CORPORATE_ACTION_COLUMNS)

        ratio_from = pd.to_numeric(df.get("ratio_from", pd.Series(index=df.index)), errors="coerce")
        ratio_to = pd.to_numeric(df.get("ratio_to", pd.Series(index=df.index)), errors="coerce")

        # ratio_from:ratio_to is written the way NSE writes it -- "split from
        # face value 10 to 2" or "bonus 1:1". Both reduce to a price factor.
        action = df["action_type"].astype(str).str.strip().str.lower()
        factor = pd.Series(1.0, index=df.index, dtype="float64")

        is_split = action.str.contains("split")
        factor[is_split] = (ratio_to / ratio_from)[is_split]

        is_bonus = action.str.contains("bonus")
        # bonus a:b -> b new shares per a held is written ratio_from=a, ratio_to=b
        factor[is_bonus] = (ratio_from / (ratio_from + ratio_to))[is_bonus]

        out = pd.DataFrame(
            {
                SYMBOL: df["symbol"].map(normalise_symbol),
                "ex_date": pd.to_datetime(df["ex_date"], errors="coerce").dt.normalize(),
                "action_type": action,
                "ratio": factor,
                "raw_details": df.get("details", ""),
                "source": self.name,
            }
        )
        out = out.dropna(subset=[SYMBOL, "ex_date"])
        out["ratio"] = out["ratio"].fillna(1.0)
        bad = out[(out["ratio"] <= 0) | (out["ratio"] > 1000)]
        if not bad.empty:
            raise DataError(
                f"corporate_actions file produced {len(bad)} implausible price "
                f"factor(s); check ratio_from / ratio_to on rows for "
                f"{bad[SYMBOL].head(5).tolist()}"
            )
        return out[CORPORATE_ACTION_COLUMNS].reset_index(drop=True)

    # =====================================================================
    # regulatory events
    # =====================================================================
    def load_regulatory_events(self, default_cooldown: int) -> pd.DataFrame:
        path = self._path("regulatory_events_file")
        df = _read_csv(path, ["symbol", "event_date"], "regulatory_events")
        if df is None:
            self._note_absent("regulatory_events", path)
            return pd.DataFrame(columns=REFERENCE_TEMPLATES["regulatory_events"])
        out = pd.DataFrame(
            {
                SYMBOL: df["symbol"].map(normalise_symbol),
                "event_date": pd.to_datetime(df["event_date"], errors="coerce").dt.normalize(),
                "cooldown_days": pd.to_numeric(
                    df.get("cooldown_days", default_cooldown), errors="coerce"
                ).fillna(default_cooldown),
                "reason": df.get("reason", "unspecified"),
            }
        )
        return out.dropna(subset=[SYMBOL, "event_date"]).reset_index(drop=True)

    # =====================================================================
    # index membership (true point-in-time, if you have it)
    # =====================================================================
    def load_index_membership(self) -> pd.DataFrame:
        """Hand-maintained historical membership with effective dates.

        This is the ONE file that can genuinely fix the survivorship problem
        for historical dates. NSE publishes index-reconstitution circulars; if
        you transcribe them here, the universe module prefers this over the
        forward-accumulating snapshots.
        """
        path = self._path("index_membership_file")
        df = _read_csv(path, ["index_name", "symbol", "effective_from"], "index_membership")
        if df is None:
            self._note_absent("index_membership", path)
            return pd.DataFrame(columns=REFERENCE_TEMPLATES["index_membership"])
        out = pd.DataFrame(
            {
                "index_name": df["index_name"].astype(str).str.strip().str.upper(),
                SYMBOL: df["symbol"].map(normalise_symbol),
                "effective_from": pd.to_datetime(
                    df["effective_from"], errors="coerce"
                ).dt.normalize(),
                "effective_to": pd.to_datetime(
                    df.get("effective_to", pd.Series(index=df.index)), errors="coerce"
                ).dt.normalize(),
            }
        )
        return out.dropna(subset=["index_name", SYMBOL, "effective_from"]).reset_index(drop=True)

    # =====================================================================
    # templates
    # =====================================================================
    def write_templates(self, overwrite: bool = False) -> List[Path]:
        """Create blank, correctly-headed CSVs so the user knows what to fill in."""
        written: List[Path] = []
        mapping = {
            "pledging": "pledging_file",
            "fundamentals": "fundamentals_file",
            "earnings_calendar": "earnings_calendar_file",
            "corporate_actions": "corporate_actions_file",
            "regulatory_events": "regulatory_events_file",
            "index_membership": "index_membership_file",
        }
        for label, attr in mapping.items():
            path = self._path(attr)
            if path.exists() and not overwrite:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(",".join(REFERENCE_TEMPLATES[label]) + "\n", encoding="utf-8")
            written.append(path)
        return written
