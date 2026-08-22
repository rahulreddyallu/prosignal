# How far the price history can honestly be extended

The engine's evidence is bounded by independent observations, not by rows. At
the 63-session horizon the price block has ~30 and the value block 11. The
obvious remedy is more history. This records how much is reachable and what it
would cost, so the question does not get reopened on the assumption that the
limit is a technical one.

## The reachable window

| source | earliest usable | why it stops there |
|---|---|---|
| NSE cash bhavcopy | **2016-01** | probed directly: HTTP 403 for 2008–2015 |
| NSE delivery (`sec_bhavdata_full`) | ~2021 on re-fetch | 0 rows returned for 2016, 2017, 2019; the store holds from 2019-07 only because it was ingested then |
| vendor statements | 2023-06 for the bulk | before that, 380–401 symbols carry statements but only **25–44 carry a share count**, and market cap needs one |

The store already reaches 2017-09. **The genuinely reachable extension is about
1.7 years, to 2016-01, and only for prices.**

## Why a longer reconstruction is not evidence

Broker and vendor APIs serve daily candles much further back — some to 2000 —
but their instrument masters carry **only currently-listed** securities and
publish no archived versions. A history rebuilt from today's master contains
exactly the companies that still exist today.

The cost is measurable. The local store was ingested progressively, so it still
holds names that traded and have since stopped:

| window | names trading | absent today | rate | annualised |
|---|---|---|---|---|
| 2017-09 → 2018-03 | 1,614 | 497 | 30.8% | **4.3%/yr** |
| 2019-01 → 2019-06 | 1,613 | 410 | 25.4% | 4.0%/yr |
| 2021-01 → 2021-06 | 1,741 | 359 | 20.6% | 4.3%/yr |
| 2023-01 → 2023-06 | 2,003 | 282 | 14.1% | 4.6%/yr |

**4.0–4.6% per year across four independent windows** — stable enough to be a
structural property of this market rather than sampling noise. Extrapolated:

| extension | share of the then-universe missing |
|---|---|
| 1.7 years (to 2016) | 7% |
| 8.9 years | 32% |
| 16 years | 50% |
| 26 years (to 2000) | **68%** |

Names that vanished from the 2017–18 window include ABAN, ADANITRANS, ADHUNIK,
ADLABS, 3IINFOTECH, 8KMILES — a list with an obvious character. They did not
leave at random, and the survivors' returns are upward-biased by exactly that
selection.

A backtest on the surviving third would report a higher n, a better Deflated
Sharpe and a lower PBO. Every one of those improvements would be an artefact.

## Rule

**Extend only as far as a point-in-time universe can be reconstructed.** For
prices that is 2016-01 with a ~7% gap that must be stated on the run. Anything
longer requires a vendor that serves delisted instruments and archived
instrument masters, which is a procurement question rather than an engineering
one.

## What would close the value block's eleven observations

A point-in-time fundamental feed with **true filing dates** and **delisted
coverage** — in this market, Prowess, Refinitiv, Capital IQ or Bloomberg. Broker
APIs do not compete in that category: their statement endpoints return
period-end labels with no publication date, which is the same position the
engine is already in.

Until such a feed exists the fundamental block stays at eleven independent
observations, and calibration, machine learning and ensembles remain
inadmissible on it. The response already in the code is the right one: a ridge
penalty of 20,000 shrinks the value coefficients hardest precisely because
their signal-to-noise is lowest.
