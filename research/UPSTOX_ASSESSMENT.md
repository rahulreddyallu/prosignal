# Upstox API — what it can and cannot fix

Assessed against the binding constraint from Phase 1: the value block has **11
independent observations** at the 63-session horizon, the price block 30. The
test is whether Upstox moves those numbers honestly, not whether it adds another
data source.

**Verdict: it does not close the fundamental gap, and the price extension it
appears to offer would be bought with 68% survivorship bias. Two of its eight
endpoints are genuinely worth wiring.**

## The fundamentals do not carry filing dates

`GET /fundamentals/:isin/income-statement` returns a `history` array of
`{value, period}` where `period` is a label like `"Mar 2025"` — **the fiscal
period end, not when it was published.** Same for balance sheet and cash flow.

That is exactly the position the engine is already in with the yfinance
statement feed, and it is why `fundamental_factors` derives availability from
the SEBI LODR deadline. Upstox would change the vendor, not the epistemics:
availability would still have to be approximated from the filing deadline.

Upstox's own announcement states the APIs *"currently return a limited financial
history"*, and the documented example shows roughly **four years** per category.
The store already holds annual statements from 2021-12 — about four and a half.

**Effect on n = 11: none.**

## The price history is real, and buying it would be a mistake

`GET /historical-candle/...` serves **daily candles from January 2000** — 26
years against the 8.9 currently held. That would take the price block from ~30
independent observations to ~100, which is the single largest number available
anywhere in this project.

It is not usable, for one reason:

> *"The BOD instrument for the next trading day will not include delisted stocks
> or expired contracts."*

and the documentation describes **no archived or historical instrument master**.
Candles can only be requested for an `instrument_key`, and the only master
available is today's. So a 26-year reconstruction would contain exactly those
companies that still exist in 2026.

### How large the bias would be, measured not assumed

The local store was ingested progressively, so it holds names that traded in
2017–2020 and have since stopped. Their disappearance rate is a direct estimate:

| window | names trading | absent today | rate | annualised |
|---|---|---|---|---|
| 2017-09 → 2018-03 | 1,614 | 497 | 30.8% | **4.3%/yr** |
| 2019-01 → 2019-06 | 1,613 | 410 | 25.4% | 4.0%/yr |
| 2021-01 → 2021-06 | 1,741 | 359 | 20.6% | 4.3%/yr |
| 2023-01 → 2023-06 | 2,003 | 282 | 14.1% | 4.6%/yr |

**4.0–4.6% per year across four independent windows.** That stability means it
is a structural rate, not sampling noise. Extrapolated:

| extension | share of the then-universe missing |
|---|---|
| 8.9 years | 32.1% |
| 16 years | 50.2% |
| **26 years (to 2000)** | **67.8%** |

Names that vanished from the 2017–18 window include ABAN, ADANITRANS, ADHUNIK,
ADLABS, 3IINFOTECH, 8KMILES — a list with an obvious character.

A backtest run on the surviving third would show a higher n, a better DSR and a
lower PBO, and every one of those improvements would be an artefact. §52 of the
mission is explicit: *do not respond to "15 holdouts" by inventing observations*.
Extending with survivors is that error wearing a longer window.

**Effect on n = 30: nominally +70, honestly zero.**

## What is worth wiring

### 1. Corporate Actions — fixes a known defect

`GET /fundamentals/:isin/corporate-actions` returns dividends, bonuses, splits,
rights, mergers, demergers, buybacks and spin-offs **with ex-dates and record
dates**, refreshed daily.

The engine currently merges NSE and yfinance actions, and that merge had a
double-application bug — NSE's `"bonus"` and yfinance's `"split_or_bonus"` for
one event both surviving the key and the ratio being applied twice. A single
authoritative source with explicit ex-dates removes the class of defect rather
than the instance.

### 2. Share Holdings — makes a NOT_TESTABLE gate testable

`GET /fundamentals/:isin/share-holdings` returns historical shareholding
patterns: promoters, FII, DII, public.

Every run currently prints:

> *promoter-pledging data is absent, so disclosure-date alignment cannot be
> enforced. The Stage 3 pledging gate reports NOT_TESTABLE — it does not pass.*

Shareholding patterns are filed under LODR within 21 days of quarter end, so the
same deadline-based availability treatment already built for statements applies
directly. This would turn a permanently-untestable gate into a testable one.

### 3. Not offered

**Delivery data.** `deliv_pct` carries the largest coefficient in the fitted
model and comes from NSE's `sec_bhavdata_full`. Upstox does not serve it. The
delivery block stays at 26 observations.

## What would actually close n = 11

A point-in-time fundamental vendor with **true filing dates** and **delisted
coverage**. In this market that means CMIE Prowess, Refinitiv, S&P Capital IQ or
Bloomberg — paid institutional feeds. Upstox is a broker API and does not
compete in that category, which is not a criticism of it.

Until such a feed exists, the fundamental block is capped at eleven independent
observations and Phases 19 and 21–23 — calibration, ML, ensembles — remain
inadmissible on it.

## Recommended sequence

1. **Corporate actions** from Upstox, replacing the two-source merge — a
   correctness win, independent of any research question
2. **Share holdings**, with LODR-deadline availability, to make the pledging
   gate testable for the first time
3. **No price extension** beyond what NSE serves natively (2016-01), and even
   that only with the ~7% survivorship gap over 1.7 years stated on the run

Order placement, GTT, exit-all-positions and every other trading endpoint are
out of scope and will not be wired. This engine issues opinions; it does not
send orders.
