# Data Sources — audit of 2026-08-17, re-probed 2026-09-04

Every endpoint below was **probed live on 2026-08-17**, not taken from
documentation. Response structures are quoted from actual returns. The point of
this file is that nobody re-probes: if a capability is listed dead here, it was
tested and found dead; if it is listed working, the exact fields are recorded.

The organising question throughout is **point-in-time integrity**: not "can I
get this number" but "can I get it *as it was known on the decision date*". A
fundamentals feed without a filing date is not a point-in-time source, however
accurate its numbers, because using a figure before it was public is lookahead
and lookahead is invisible in a backtest.

> [!IMPORTANT]
> **Re-probed 2026-09-04 (v10 Pass 1). One capability recorded here is GONE and
> one is far better than recorded. See "Re-probe, 2026-09-04" immediately below
> before trusting the 2026-08-17 tables.**
>
> The August audit's own principle — "nobody re-probes" — is the reason this
> section exists rather than an edit in place. An endpoint that answered three
> weeks ago is not evidence that it answers today, and one of these had already
> stopped answering in the form the engine was built to call.

---

## Re-probe, 2026-09-04

Same client, same cookie handshake, same host. The warm-up still works and the
JSON API is still reachable from this machine.

| Endpoint | 2026-08-17 recorded | 2026-09-04 measured | Verdict |
|---|---|---|---|
| `corporates-financial-results` **by date window** | 91 rows / 12 months | **0 rows at every window width tried** (1, 3, 8, 12 months) | **GONE** |
| `corporates-financial-results` **by symbol** | not recorded | **130 rows for RELIANCE**, `period=Quarterly` | **NEW, and much better** |
| `corporates-pit` | 9,998 rows / 12 months | 9,354 rows / 12 months; **0 rows at 3 months** | unchanged, quirk confirmed |
| `corporate-share-holdings-master` | 2,286 rows | 20–22 rows per symbol, symbol-keyed | working |
| `corporate-sast-reg29` | 2,755 rows / 6 months | 447 rows / 1 month | working |
| `corporate-announcements` | 9.8 MB / 17 days | 1,539 rows / 4 days | working |

### The date-window form of `corporates-financial-results` is dead

This matters because it is the form the engine was built to call and the one
Pass 1 was planned around. It returns a well-formed empty result at **every**
window width — not an error, not a 404, an empty list. Under the convention
this file already sets out, an empty window is **UNKNOWN**; four widths all
empty while the same endpoint answers richly by symbol is not unknown, it is a
changed interface.

### The by-symbol form is a large upgrade, not a workaround

```
/api/corporates-financial-results?index=equities&symbol=<SYM>&period=Quarterly
```

Measured across RELIANCE, TCS, HDFCBANK, INFY, ITC, SBIN, LT, KTKBANK:

| | |
|---|---|
| rows per symbol | 95 – 162 |
| period-end span | **2005-03-31 → 2024-12-31**, identical on all eight |
| `filingDate` present | yes, on every row |
| `xbrl` link present | **100% of rows** (direct nsearchives Ind-AS URL) |
| `resultDetailedDataLink` | null on 55 of RELIANCE's 130 rows — **use `xbrl`, not this** |
| throughput | ~0.4 s/symbol → ~5 minutes for a 750-name universe |
| filing lag (RELIANCE, n=122) | min 13, median 22, p90 74, max 457 days |

Two consequences worth stating plainly:

1. **Twenty years of point-in-time quarterly history is reachable, not five.**
   The store's `fundamentals.parquet` currently holds 3,504 rows over
   2019-11 → 2025-03 for 186 symbols. This path reaches 2005 for the whole
   universe. That is the single biggest change to what the value and quality
   themes could be built on — both were dropped for coverage, on the finding
   that "balance-sheet data begins 2023 and the median training date has ZERO
   names with a book value".

2. **It stops at 2024-12-31, on every symbol.** That is a property of the
   endpoint, not of the companies — all eight symbols end on the same date. So
   this feed is ~20 months stale at the front and **cannot serve the recent
   panel**. Recent quarters remain `NOT_TESTABLE` from this path and must be
   sourced elsewhere before any factor built on it can be computed for a live
   date. Nothing here may be interpolated forward.

### Shareholding lag, re-measured

The August audit measured period-end → disclosure across 800 filings at
min 20 / median 21 / p90 21 / max 45. Re-measured across 169 filings on eight
symbols: **min 0, median 18, p90 21, max 115**. The median and p90 agree; the
tails do not, and the `min 0` — a broadcast stamped the same day as the period
it describes — is not credible as a disclosure lag and should be treated as a
data-quality flag on that row rather than as a fast filer.

This feed **is** current: period ends run to **2026-06-30**.

---

## Correction to a Chunk 1 finding

Chunk 1 recorded: *"`www.nseindia.com` (JSON API) — 403, bot-shielded."*

**That is wrong, and it cost us real capability.** The JSON API returns **200**
once a cookie session is established. The mistake was probing it without first
loading a page to obtain cookies.

The working pattern:

1. `GET https://www.nseindia.com/companies-listing/<relevant-page>` with a
   browser `User-Agent`. This returns 200 and sets `_abck`, `bm_sz`, `nsit`.
2. Reuse that cookie jar for `GET /api/<endpoint>` with `Referer` set to the
   page from step 1.

Note the homepage `https://www.nseindia.com/` itself **does** return 403 — which
is likely how the original wrong conclusion was reached. The specific listing
pages do not.

Everything in the "working" table below was unavailable to the engine because
of that one wrong note.

### The root cause was a single config value

Chunk 1 had **already built** `NseJsonSession` in `providers/http.py`, complete
with the cookie handshake, browser `Sec-Fetch-*` headers, and soft-failure
semantics. It was correct code. It never worked because
`providers.nse_json_api.warmup_path` was set to `"/"` — the one path that
403s.

Fixed on 2026-08-17 to
`/companies-listing/corporate-filings-announcements`. Verified through the
engine's own client: warm-up now succeeds and four endpoints return data. No
new provider code was needed to prove it.

### Integration caveats measured on the live API

- **Row cap around 10,000.** `corporates-pit` returned exactly 9,998 rows for a
  12-month window. Any ingester must page by date window rather than requesting
  a long span and trusting the result is complete.
- **Short windows can return zero.** The same endpoint returned 0 rows for a
  3-month window that certainly contains filings, then 9,998 for 12 months. The
  date filter is not simply inclusive-between; a window that comes back empty
  must be treated as *unknown*, never as "no events". Getting this wrong would
  silently turn "we could not check" into "the check passed", which is the
  exact failure mode the `NOT_TESTABLE` convention exists to prevent.
- **Warm-up page does not need to match the endpoint.** Both the announcements
  and insider-trading listing pages yield cookies that work for any endpoint,
  so one warm-up per session is enough.

---

## Working — NSE JSON API (cookie session required)

| Endpoint | Volume returned | Point-in-time field | Unlocks |
|---|---|---|---|
| `/api/corporates-pit` | **9,998 rows** / 12 months | `intimDt`, `date` | Stage 5 `insider_activity` |
| `/api/corporate-announcements` | **9.8 MB** / 17 days | `an_dt` | Stage 3 `regulatory_cooldown`, Stage 5 `regulatory_shock` |
| `/api/corporate-sast-reg29` | **2,755 rows** / 6 months | `timestamp`, `time` | Bulk stake changes (>5% disclosures) |
| `/api/corporate-share-holdings-master` | **2,286 rows** | `broadcastDate`, `date` | Promoter holding %, PIT-correct |
| `/api/corporates-financial-results` | 91 rows / 12 months | **`filingDate`**, `broadCastDate` | Stage 4 `quality` — the PIT fundamentals path |

### Response structures, as actually returned

**`corporates-pit`** — insider trading (SEBI PIT regulations):
```
symbol, company, acqName, personCategory, acqMode, secType,
befAcqSharesNo, befAcqSharesPer, afterAcqSharesNo, afterAcqSharesPer,
buyQuantity, buyValue, sellquantity, sellValue,
acqfromDt, acqtoDt,          <- when the transaction happened
date, intimDt                <- when it became PUBLIC  (use this one)
```
Sample: `{"acqMode":"ESOP","acqName":"Shantanu Lath","personCategory":...,
"befAcqSharesNo":"50000","afterAcqSharesNo":"120000",...}`

**`corporate-share-holdings-master`** — quarterly shareholding:
```
symbol, isin, name, pr_and_prgrp, public_val, employeeTrusts,
date,              <- PERIOD END      e.g. "30-JUN-2026"
broadcastDate,     <- PUBLIC DISCLOSURE e.g. "21-JUL-2026 15:44:00"
submissionDate, revisedDate, revisionDate, xbrl, remarksWeb
```

**`corporates-financial-results`** — quarterly results:
```
symbol, isin, companyName, audited, consolidated, indAs, period,
fromDate, toDate, financialYear, relatingTo,
filingDate,                  <- "06-Aug-2026 14:07"
broadCastDate,               <- "06-Aug-2026 14:07:24"
resultDetailedDataLink,      <- the actual line items live behind this
xbrl
```

### The measured lookahead window

Across 800 shareholding filings, period end → public disclosure:

| min | median | p90 | max |
|---|---|---|---|
| 20 days | **21 days** | 21 days | 45 days |

This is not a theoretical concern. Any feed that hands you Q1 figures keyed to
`30-JUN` without also telling you they became public on `21-JUL` will leak
**three weeks of lookahead** into every backtest, and the backtest will look
better for it. `broadcastDate` is what makes the difference enforceable.

---

## Dead — do not retry

| Endpoint | Result |
|---|---|
| `/api/corporate-pledgedata` | **0 rows, always.** Tested with and without cookies, with and without date parameters, over 1-month and 12-month windows. Returns a well-formed `{"comNameList":[],"data":[]}` — the endpoint exists and answers, it simply has no content. |
| `/api/corporate-sast-reg31` | HTTP 404 |
| `nsearchives.../shareholding_patterns.csv` | HTTP 404 |

**Promoter pledging** is therefore not available from any single NSE JSON feed.
It is *not* impossible: SEBI LODR requires "shares pledged or otherwise
encumbered" in the shareholding pattern, and the `xbrl` link on every
`corporate-share-holdings-master` row points at the filing that contains it.
Extracting it means one XBRL fetch per company per quarter (~200 × 4 = 800
fetches/year for the NIFTY 200) — entirely feasible against the existing cache
layer, but a real piece of work rather than a field read.

**Decision: the pledging gate stays.** It currently reports `NOT_TESTABLE`,
which is honest and correct, and the CSV drop-in still works for anyone who
sources the data. Deleting it would discard a capability with a proven path to
data.

---

## Upstox Developer API — assessment

**Company Fundamentals API** (8 endpoints, ISIN-keyed): Company Profile,
Balance Sheet, Cash Flow, Income Statement, Share Holdings, Key Ratios,
Corporate Actions, Competitors.

**Licensing** (from Upstox staff on the developer forum): no licensing,
redistribution, or attribution requirements currently; responses may be cached
"for your application's internal processing"; APIs are "intended for
consumption within your own applications". Supported via an Analytics Token
with extended validity. Nothing here obstructs this engine's use.

**Historical candles**: minutes/hours from January 2022; days/weeks/months from
January 2000.

### Verdict: good API, wrong tool for the point-in-time job

| Need | Upstox | Assessment |
|---|---|---|
| Historical OHLCV | days/weeks/months from 2000 | Genuinely good — but NSE bhavcopy is already the authoritative primary source, free, and needs no auth. No reason to switch. |
| **PIT fundamentals** | Upstox confirmed *"APIs currently return a limited financial history"*; a developer reported **only the latest 4 annual periods**, and Upstox confirmed `full_statement` returns annual data regardless of the `time_period` parameter | **Unsuitable.** Four annual periods cannot support a 10-year CPCV, and no filing-date field is documented — so restatements and disclosure lag are both invisible. |
| Share holdings | quarterly promoter/FII/DII/public | Duplicates what NSE gives us *with* `broadcastDate`. NSE is better for this specific reason. |
| Key ratios | P/E, P/B, ROE, ROCE, D/E vs sector | Current-vintage only. Fine for a live display read; unusable as a validated factor input. |

**Where Upstox would genuinely add value later**: intraday candles (NSE
archives are end-of-day only) and live quotes for execution-time checks.
Neither is needed before Chunk 9. If you want to wire it up, the credentials
belong in an environment variable or a gitignored file that you populate — I
should never handle the token value itself.

---

## Net effect on the engine

### Removed

**`estimate_revision_momentum`** (Stage 4 factor). The only gap in this audit
with *no path to data at all*. It needs timestamped analyst consensus
estimates; I/B/E/S, Refinitiv and Bloomberg are the only real sources and all
are paid and licence-restricted. Critically, unlike every other gap here, it
cannot be derived — a changed analyst opinion leaves no trace in price, volume,
or filings.

It shipped disabled with a weight locked at zero, which meant config, a schema
class, and a validator all existed to hold a factor that could never fire.
Carrying that scaffolding is worse than not having it, because it implies the
capability is one config change away. It is not.

`tests/test_config.py::test_estimate_revision_factor_cannot_be_reintroduced`
now proves the removal is *enforced*: `extra="forbid"` rejects the key, so
nobody can reinstate it and quietly approximate it from an untimestamped
source.

### Kept, with a proven path to data

| Capability | Path | State |
|---|---|---|
| `insider_activity` (Stage 5) | `corporates-pit` | Endpoint proven, fields recorded, PIT via `intimDt` |
| `regulatory_shock` (Stage 5) | `corporate-announcements` | Endpoint proven, PIT via `an_dt` |
| `regulatory_cooldown` (Stage 3) | `corporate-announcements` | Same feed |
| `quality` (Stage 4) | `corporates-financial-results` → `resultDetailedDataLink` | Endpoint and `filingDate` proven; line-item extraction still to build |
| `pledging` (Stage 3) | shareholding `xbrl` attachment | Path proven, extraction still to build. Reports `NOT_TESTABLE` until then — never `PASS` |

None of these are guesses. Each has a probed endpoint and a recorded response
structure above.

---

## Standing caveat on these endpoints

The `www.nseindia.com/api/*` endpoints are **undocumented and unofficial**.
They exist to serve NSE's own website, carry no stability guarantee, and sit
behind Akamai bot protection that can tighten without notice. The engine must
therefore treat every one of them as *optional*: a failure degrades a check to
`NOT_TESTABLE` and is reported on the card. None may become a required feed,
because a required feed that NSE can revoke silently is an outage waiting to
happen.

The `nsearchives.nseindia.com` archive files remain the authoritative primary
source — they need no cookies, no auth, and have been stable throughout.
