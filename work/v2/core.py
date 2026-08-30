"""Ranking, costs, portfolio simulation and purged cross-validation.

Everything the search loop needs. `load_train()` is the only data entry point
here; the sealed holdout has no loader in this module by design.
"""
from __future__ import annotations
import numpy as np, pandas as pd, json

CACHE = "/home/claude/psr/cache"
MIN_SECTOR_NAMES = 12
SESSIONS_PER_YEAR = 252.0
STEP_SESSIONS = 5


def load_train() -> pd.DataFrame:
    return pd.read_parquet(f"{CACHE}/TRAIN.parquet")


def seal() -> dict:
    return json.load(open(f"{CACHE}/SEAL.json"))


# ---------------------------------------------------------------- ranking
def xs_rank(v: np.ndarray) -> np.ndarray:
    """Rank to [-1, 1] within a cross-section, NaN preserved."""
    ok = np.isfinite(v)
    out = np.full(v.shape, np.nan)
    n = ok.sum()
    if n < 2:
        return out
    r = pd.Series(v[ok]).rank(pct=True).to_numpy()
    out[ok] = (r - 0.5) * 2.0
    return out


def sector_neutral_ranks(df: pd.DataFrame, cols: list, sector_col="sector") -> pd.DataFrame:
    """Rank within sector where the bucket is big enough, else within ONE
    residual group per date. One normalisation everywhere -- never a mix of
    within-sector and whole-universe ranks in the same column."""
    d = df[["date", sector_col]].copy()
    cnt = d.groupby(["date", sector_col], observed=True)[sector_col].transform("size")
    big = (cnt >= MIN_SECTOR_NAMES) & (d[sector_col].astype(str) != "UNCLASSIFIED")
    grp = np.where(big, d[sector_col].astype(str), "__RESID__")
    key = pd.Series(d["date"].astype("int64").astype(str) + "|" + grp, index=df.index)
    r = df[cols].groupby(key, observed=True).rank(pct=True, na_option="keep")
    out = ((r - 0.5) * 2.0).astype("float32")
    out.columns = [c + "_r" for c in cols]
    return out


# ---------------------------------------------------------------- costs
class Costs:
    """Indian cash-equity delivery costs, config rates from parameters.yaml."""
    STT_BUY = 0.001; STT_SELL = 0.001
    STAMP_BUY = 0.00015
    EXCH = 0.0000297; SEBI = 0.000001
    GST = 0.18
    BROKERAGE_FLAT = 20.0
    DP_SELL = 15.93
    IMPACT_COEF = 0.10; IMPACT_EXP = 0.5; HALF_SPREAD_BPS = 5.0

    @classmethod
    def one_way_bps(cls, value_inr: float, adtv_inr, side: str) -> np.ndarray:
        v = np.asarray(value_inr, dtype="float64")
        a = np.asarray(adtv_inr, dtype="float64")
        part = np.where((a > 0) & np.isfinite(a), v / np.maximum(a, 1.0), 0.01)
        impact = cls.IMPACT_COEF * np.power(part, cls.IMPACT_EXP) * 1e4
        spread = cls.HALF_SPREAD_BPS
        stt = (cls.STT_BUY if side == "buy" else cls.STT_SELL) * 1e4
        stamp = cls.STAMP_BUY * 1e4 if side == "buy" else 0.0
        exch = cls.EXCH * 1e4; sebi = cls.SEBI * 1e4
        brok = np.where(v > 0, cls.BROKERAGE_FLAT / np.maximum(v, 1.0) * 1e4, 0.0)
        gst = (brok + exch + sebi) * cls.GST
        dp = np.where(side == "sell", cls.DP_SELL / np.maximum(v, 1.0) * 1e4, 0.0)
        return impact + spread + stt + stamp + exch + sebi + brok + gst + dp


def prep_date(g) -> dict:
    g = g.sort_values("score", ascending=False)
    order = g["symbol"].to_numpy()
    return {"order": order,
            "rank": {s: i + 1 for i, s in enumerate(order)},
            "ret": dict(zip(order, g["y5"].to_numpy("float64"))),
            "adtv": dict(zip(order, g["adtv"].to_numpy("float64"))),
            "sect": dict(zip(order, g["sector"].to_numpy())),
            "score": dict(zip(order, g["score"].to_numpy("float64"))),
            "ivol": (dict(zip(order, 1.0 / np.maximum(
                g["atr_pct"].to_numpy("float64"), 1e-4)))
                if "atr_pct" in g else {}),
            "ok": (dict(zip(order, g["name_ok"].to_numpy()))
                   if "name_ok" in g else {}),
            "bench": float(g["b5"].iloc[0]) if np.isfinite(g["b5"].iloc[0]) else 0.0}


def prepare(scored: pd.DataFrame) -> dict:
    """Pre-slice a scored frame once so a parameter sweep does not pay the
    pandas cost per configuration."""
    cols = ["date", "symbol", "score", "y5", "b5", "adtv", "sector"]
    for extra in ("atr_pct", "name_ok"):
        if extra in scored:
            cols.append(extra)
    d = scored[cols].dropna(subset=["y5", "score"])
    dates = np.array(sorted(d["date"].unique()))
    return {"by": {k: prep_date(g) for k, g in d.groupby("date", sort=False)},
            "dates": dates}


# ---------------------------------------------------------------- portfolio
def simulate(scored: pd.DataFrame, *, slots=8, entry_rank=8, exit_rank=16,
             rebalance_every=1, capital=1_000_000.0, weighting="equal",
             score_floor=None, max_per_sector=2, regime_ok=None,
             stop_atr=None, cash_rate=0.0, require_name_ok=False) -> dict:
    """Rank-band long book with hysteresis, filled at next-session VWAP.

    A name is opened only while its rank is inside `entry_rank` AND, when a
    floor is set, its score clears it. Slots left unfilled hold CASH at
    `cash_rate` (0 by default, so sitting out is never rewarded by an assumed
    yield). NO TRADE is therefore a reachable, unpenalised state.
    """
    if isinstance(scored, dict):
        by, dates = scored["by"], scored["dates"]
    else:
        req = ["date", "symbol", "score", "y5", "b5", "adtv", "sector"]
        d = scored[req + (["atr_pct"] if "atr_pct" in scored else [])].dropna(subset=["y5", "score"])
        dates = np.array(sorted(d["date"].unique()))
        by = {k: prep_date(g) for k, g in d.groupby("date", sort=False)}
    held: dict = {}
    dwell: dict = {}
    closed_holds: list = []
    eq = 1.0
    curve, bench_curve, turn_hist, ncash = [], [], [], []
    bench_eq = 1.0
    slot_value = capital / slots
    for t, dt_ in enumerate(dates):
        P = by[dt_]
        order, rank, ret, adtv, sect, score, bench, ivol = (
            P["order"], P["rank"], P["ret"], P["adtv"], P["sect"],
            P["score"], P["bench"], P["ivol"])
        nameok = P.get("ok", {})

        do_rebal = (t % rebalance_every == 0)
        turnover_bps_cost = 0.0
        if do_rebal:
            gate_open = True if regime_ok is None else bool(regime_ok.get(dt_, True))
            keep = {}
            for s in held:
                r = rank.get(s)
                if r is not None and r <= exit_rank and (score_floor is None or score.get(s, -9) >= score_floor):
                    keep[s] = held[s]
            cands = []
            if gate_open:
                per_sec = {}
                for s in keep:
                    per_sec[sect.get(s)] = per_sec.get(sect.get(s), 0) + 1
                for s in order:
                    if s in keep:
                        continue
                    if rank[s] > entry_rank:
                        break
                    if score_floor is not None and score.get(s, -9) < score_floor:
                        continue
                    if require_name_ok and not bool(nameok.get(s, True)):
                        continue
                    sc = sect.get(s)
                    if max_per_sector and per_sec.get(sc, 0) >= max_per_sector:
                        continue
                    cands.append(s)
                    per_sec[sc] = per_sec.get(sc, 0) + 1
                    if len(keep) + len(cands) >= slots:
                        break
            new = dict(keep)
            for s in cands:
                if len(new) >= slots:
                    break
                new[s] = 1.0
            sold = [s for s in held if s not in new]
            bought = [s for s in new if s not in held]
            n_trades = len(sold) + len(bought)
            if n_trades:
                sell_bps = Costs.one_way_bps(slot_value,
                                             np.array([adtv.get(s, np.nan) for s in sold]) if sold else np.array([]),
                                             "sell")
                buy_bps = Costs.one_way_bps(slot_value,
                                            np.array([adtv.get(s, np.nan) for s in bought]) if bought else np.array([]),
                                            "buy")
                cost_inr = (np.nansum(sell_bps) + np.nansum(buy_bps)) / 1e4 * slot_value
                turnover_bps_cost = cost_inr / capital
            for s in sold:
                closed_holds.append(dwell.pop(s, 1))
            for s in new:
                dwell[s] = dwell.get(s, 0)
            held = new
        # weights
        if held:
            if weighting == "equal":
                w = {s: 1.0 / slots for s in held}
            elif weighting == "invvol":
                iv = {s: float(ivol.get(s, 0.0)) for s in held}
                tot = sum(iv.values()) or 1.0
                cap_each = 1.0 / slots * 1.6
                w = {s: min(iv[s] / tot * (len(held) / slots), cap_each) for s in held}
            else:
                w = {s: 1.0 / slots for s in held}
        else:
            w = {}
        r_book = 0.0
        for s, wt in w.items():
            rr = ret.get(s)
            if rr is None or not np.isfinite(rr):
                rr = 0.0
            r_book += wt * float(rr)
        cash_w = max(0.0, 1.0 - sum(w.values()))
        r_book += cash_w * cash_rate
        r_book -= turnover_bps_cost
        for s in held:
            dwell[s] = dwell.get(s, 0) + 1
        eq *= (1.0 + r_book)
        bench_eq *= (1.0 + bench)
        curve.append(eq); bench_curve.append(bench_eq)
        turn_hist.append(turnover_bps_cost); ncash.append(cash_w)
    curve = np.array(curve); bench_curve = np.array(bench_curve)
    rets = np.diff(np.log(np.maximum(curve, 1e-9)))
    per_yr = SESSIONS_PER_YEAR / STEP_SESSIONS
    n = len(curve)
    ann = curve[-1] ** (per_yr / max(n, 1)) - 1.0 if n else np.nan
    bann = bench_curve[-1] ** (per_yr / max(n, 1)) - 1.0 if n else np.nan
    simple = np.diff(curve) / curve[:-1] if n > 1 else np.array([0.0])
    sharpe = (simple.mean() / (simple.std(ddof=1) + 1e-12)) * np.sqrt(per_yr) if n > 2 else np.nan
    dd = float((curve / np.maximum.accumulate(curve) - 1.0).min()) if n else np.nan
    closed_holds.extend(dwell.values())
    hold_sessions = (np.array(closed_holds) * STEP_SESSIONS) if closed_holds else np.array([np.nan])
    return {"median_hold_sessions": float(np.nanmedian(hold_sessions)),
            "mean_hold_sessions": float(np.nanmean(hold_sessions)),
            "n_closed": len(closed_holds),
            "ann": float(ann), "bench_ann": float(bann), "sharpe": float(sharpe),
            "maxdd": dd, "final": float(curve[-1]), "n_periods": n,
            "cost_drag_ann": float(np.mean(turn_hist) * per_yr),
            "cash_share": float(np.mean(ncash)),
            "curve": curve, "bench_curve": bench_curve}


# ---------------------------------------------------------------- IC
def rank_ic(df: pd.DataFrame, score_col: str, y_col: str) -> tuple:
    ics = []
    for _, g in df.groupby("date", sort=False):
        a = g[score_col].to_numpy("float64"); b = g[y_col].to_numpy("float64")
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 30:
            continue
        ra = pd.Series(a[m]).rank().to_numpy(); rb = pd.Series(b[m]).rank().to_numpy()
        if ra.std() < 1e-9 or rb.std() < 1e-9:
            continue
        ics.append(float(np.corrcoef(ra, rb)[0, 1]))
    ics = np.array(ics)
    if len(ics) < 3:
        return np.nan, np.nan, len(ics)
    return float(ics.mean()), float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))), len(ics)


def topk_excess(df: pd.DataFrame, score_col: str, y_col: str, k: int = 8) -> tuple:
    ex = []
    for _, g in df.groupby("date", sort=False):
        g = g.dropna(subset=[score_col, y_col])
        if len(g) < 50:
            continue
        top = g.nlargest(k, score_col)[y_col].mean()
        ex.append(float(top - g[y_col].mean()))
    ex = np.array(ex)
    if len(ex) < 3:
        return np.nan, np.nan, len(ex)
    return float(ex.mean()), float(ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))), len(ex)


# ---------------------------------------------------------------- CV
def purged_folds(dates: np.ndarray, n_folds: int = 5, horizon_sessions: int = 21,
                 embargo_sessions: int = 21, step_sessions: int = STEP_SESSIONS):
    """Contiguous test blocks with the label window purged out of training and
    an embargo after it. Purge is measured in SIGNAL STEPS, not sessions."""
    d = np.array(sorted(dates))
    gap = int(np.ceil((horizon_sessions + embargo_sessions) / step_sessions))
    bounds = np.linspace(0, len(d), n_folds + 1).astype(int)
    for f in range(n_folds):
        lo, hi = bounds[f], bounds[f + 1]
        test = d[lo:hi]
        tr_lo = d[: max(lo - gap, 0)]
        tr_hi = d[min(hi + gap, len(d)):]
        train = np.concatenate([tr_lo, tr_hi])
        if len(train) < 20 or len(test) < 5:
            continue
        yield train, test
