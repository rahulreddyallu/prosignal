# Pro Stock Signal BOT

**India solo-quant decision-support signal engine — NSE equities, medium horizon.**

> Decision-support tool. Not financial advice. No trades are placed automatically.
> This repository contains no order-routing code, and `api.allow_order_placement`
> is a hard interlock that the config loader refuses to let you turn on.

It ranks and screens NSE-listed equities through a fixed 9-stage pipeline and
returns either a small set of high-conviction candidates or an explicit
`NO TRADE` state. It reasons in **bands, gates and explicit hypotheses** — never
in fabricated precision. There is no invented "73.4% confidence" anywhere,
and no backtested Sharpe ratio that was not actually computed on real data.

---

## The one file you edit

**`config/parameters.yaml`.**

Every threshold, weight, lookback, cost rate and switch in the engine lives
there. No core module hardcodes a magic number — if you find one, that is a bug.

Each parameter carries its own provenance:

| status | meaning |
|---|---|
| `UNVALIDATED` | A hypothesis. Never been through CPCV on point-in-time India data. The engine uses it *and tells you so on every output.* |
| `VALIDATED` | Promoted after a real CPCV/PBO/DSR run. The loader **refuses** this status unless you also supply `validated_by` (ledger trial id) and `validated_on`. |
| `STATUTORY` | Fixed by SEBI / the exchange / the tax code. Verify against the live circular. |
| `STRUCTURAL` | Definitional — changing it changes what the factor *is* (12-1 momentum is 252/21 by academic construction). |
| `OPERATIONAL` | Your own business constraint: capital, broker fees, risk appetite. |

As shipped: **182 parameters, 132 of them `UNVALIDATED`.** That number is
supposed to be large right now. It is the honest state of a system whose
validation harness has not been run yet.

The loader is strict on purpose:

- an unknown key is a hard error, never a silent fallback to a hidden default;
- a value outside its own declared `search_range` is a hard error;
- cross-section contradictions are hard errors (history shorter than the longest
  lookback, CPCV purge shorter than the label horizon, and so on).

A fat-fingered edit fails at startup rather than reaching a live decision.

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt && .venv/bin/python -m pip install -e .
```

macOS system Python ships with LibreSSL, which urllib3 v2 rejects. If you hit
TLS warnings:

```bash
.venv/bin/python -m pip install "urllib3<2"
```

Check the config loads and see every parameter with its status:

```bash
.venv/bin/prosignal config validate
```

```bash
.venv/bin/prosignal config show --unvalidated-only
```

Create the blank reference CSVs for the feeds no free source supplies:

```bash
.venv/bin/prosignal config templates
```

Pull data (first run takes a few minutes; later runs are served from the local
HTTP cache in seconds):

```bash
.venv/bin/prosignal data ingest --full
```

Inspect what you have, and run the integrity checks:

```bash
.venv/bin/prosignal data status
```

```bash
.venv/bin/prosignal data check
```

Run the test suite:

```bash
.venv/bin/python -m pytest tests/ -q
```

---

## Where the data actually comes from

Every endpoint below was probed against the live hosts during the build.

| Feed | Source | Status |
|---|---|---|
| Daily OHLCV, all NSE cash names | NSE bhavcopy (UDiFF ≥ 2024-07-08, legacy before) | working |
| Delivery quantity & percentage | NSE `sec_bhavdata_full` | working |
| Nifty 50 / 200 / 500, **every sector index, and India VIX** | NSE `ind_close_all` — one file per session | working |
| Index constituents + sector labels | NSE `ind_nifty<N>list.csv` | working (current vintage only) |
| Listing dates, ISIN, all symbols | NSE `EQUITY_L.csv` | working |
| F&O open interest | NSE F&O bhavcopy | working |
| Corporate-action ratios | Yahoo (`Stock Splits` captures Indian bonus issues) + your CSV | working |
| Scheduled earnings dates | Yahoo + your CSV | working (estimates, not board notices) |
| Second price source for cross-checking | Yahoo Finance | working |
| **Promoter pledging** | **your CSV only** | **no reliable free source → reports `NOT_TESTABLE`** |
| **Point-in-time fundamentals** | **your CSV only** | **no reliable free source → quality factor is dropped and said so** |

Two things worth knowing before you trust anything:

**`www.nseindia.com`'s JSON API is bot-shielded** and returned 403 from the
build machine while the archive hosts returned 200. It is implemented as
best-effort, and *no required feed depends on it*.

**A missing feed is never a passing check.** If pledging data is absent, the
gate reports `NOT_TESTABLE` and that line appears on the output. An engine that
treated "I could not check" as "it's fine" would be worse than one with no
check at all, because it launders an unknown into a reassurance.

---

## The point-in-time problem, stated plainly

The research program is blunt that nothing downstream is trustworthy without a
genuine point-in-time dataset. NSE publishes only *today's* constituent list, so
there is no free way to retroactively fix history. What this engine does instead
is refuse to hide it:

1. It snapshots the live constituent list, dated, on **every run** — so real
   point-in-time membership accumulates going forward from today.
2. If you ask for a date before your earliest snapshot, it sets
   `survivorship_risk = True` on the manifest and says so in the output.
3. `universe.pre_snapshot_policy: halt` makes that condition **refuse to run**.
   Switch it to `halt` the moment you start backtesting. `flag` is only
   defensible for live/forward runs, where today's list genuinely *is* the
   point-in-time list.
4. `config/reference/index_membership.csv` is the real fix. Transcribe NSE's
   index-reconstitution circulars into it and the resolver prefers it over
   snapshots.

Known, acknowledged gap: **sector labels are current-vintage, not historical.**
A company reclassified since your window inherits its new sector for old dates.
The only clean fix is a paid point-in-time classification feed dropped into the
CSV importer.

Also live, and already earning its keep: the **unexplained-jump detector**. On
the very first real ingest it flagged `IVZINNIFTY` moving 0.1007× overnight
against a clean 10:1 split factor, with no corporate action on file. Left alone,
that reads as a −90% single-session return and poisons a 12-1 momentum score for
a full year.

---

## Layout

```
config/
  parameters.yaml           <- THE FILE YOU EDIT
  reference/                <- your CSV drop-ins (pledging, fundamentals, ...)
src/prosignal/
  config/     schema.py     <- strict pydantic model of every parameter
              loader.py     <- validation, hashing, config_version stamping
  core/       contracts.py  <- Stage 0-8 input/output schemas
              calendar.py   <- NSE sessions, discovered from data not guessed
              enums.py errors.py logging.py paths.py
  data/       ingest.py     <- Stage 0 orchestrator + run manifest
              store.py      <- parquet point-in-time store, atomic + idempotent
              universe.py   <- point-in-time membership resolution
              corporate_actions.py
              providers/    <- nse_archives, yfinance, csv_import, http
  cli.py
tests/
data/                       <- everything the engine writes (git-ignored)
```

---

## Design rules the code enforces

1. **Hard rejects fire before score penalties.** Liquidity, data-integrity and
   staleness are binary gates, never blended into a continuous score.
2. **Eligibility filtering happens before scoring**, not after.
3. **`NO TRADE` is a first-class output**, never an empty array or an error. When
   it fires it must still report which candidates came closest and which gate
   stopped each.
4. **Every run is logged** to the append-only research ledger. An unlogged run
   corrupts the Deflated Sharpe trial count and therefore every subsequent
   statistical claim, which is why the ledger writer is fatal-on-failure.
5. **No forward-fill across sessions, anywhere.** Gaps stay gaps so the
   continuity check can see them.
6. **Equal-weighting is the default**, not rank-IC weighting. In-sample IC gains
   are usually overfitting.

---

## Status

See [`BUILD_LOG.md`](BUILD_LOG.md) for what is built, what is not, and exactly
where to resume.

**Built:** config system, stage contracts, trading calendar, point-in-time store,
all data providers, Stage 0 ingestion + manifest, corporate-action engine, CLI,
91 tests.

**Not built yet:** Stages 1–8, research ledger, API, webapp, CPCV/PBO/DSR harness.

---

## Research basis

The engine's factor choices, evidence tiers and refusals trace to specific
sources. The main load-bearing ones:

- Jegadeesh & Titman (1993); Rouwenhorst (1998); Asness, Moskowitz & Pedersen
  (2013) — cross-sectional momentum.
- Daniel & Moskowitz (2016); Barroso & Santa-Clara (2015) — momentum crashes and
  volatility scaling.
- Asness, Frazzini & Pedersen — Quality Minus Junk; quality as a crash
  stabiliser rather than co-equal alpha.
- Kaminski & Lo (2014) — stops add value under positive serial correlation,
  which a momentum-anchored book satisfies. They establish *when* stops help,
  not a multiplier; anyone quoting "2× ATR is optimal" is quoting a blog.
- Kalia (2024) — promoter pledging predicts crash risk, so it is a risk-exclusion
  filter and never a return signal.
- Thenmozhi & Chandra (2013); G.C. & Kothari (2016) — India VIX is asymmetric;
  a rising-VIX flag is more reliable than a falling-VIX all-clear.
- Bailey, Borwein, López de Prado & Zhu (2014); Bailey & López de Prado (2014);
  Arian, Norouzi & Seco (2024) — PBO, Deflated Sharpe Ratio, CPCV.
- Harvey, Liu & Zhu (2016) — t > 3.0, not 2.0.
- Sharma, Subramaniam & Sehgal (2021) — India-specific caveat: value and momentum
  were largely explained away by risk models on NSE 500, 2005–2016.
- India PEAD evidence is **contradictory** (Harshita 2018 vs the 2014–2018 ERC
  study). The engine therefore locks PEAD at zero weight rather than picking the
  friendlier study.
