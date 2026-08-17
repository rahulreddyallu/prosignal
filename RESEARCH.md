# Research → Ideal System → Gap Analysis

**Scope note, stated up front:** this is a *bounded* research pass — four
targeted literature searches, not an exhaustive review. Every citation below is
something I actually retrieved. Where I could not verify a claim, it says
UNVERIFIED rather than acquiring a plausible-sounding reference. Evidence
quality is tiered explicitly, because on this topic the tier matters more than
the count.

---

## 1. WHAT THE RESEARCH SAYS

### 1.1 The finding that should reshape the design

**Momentum returns are lower in emerging markets, and transaction costs are
higher.** Korajczyk & Sadka (2004) find that wider bid-ask spreads and the
price impact of illiquid stocks can **erode momentum profits entirely**, and
that strategies focused on **large-cap, liquid** names are the ones most likely
to survive implementation
([Are Momentum Profits Robust to Trading Costs?](https://www.researchgate.net/publication/4769357_Are_Momentum_Profits_Robust_to_Trading_Costs);
[Revisiting momentum profits in emerging markets](https://www.sciencedirect.com/science/article/abs/pii/S0927538X20306983)).

This is a double squeeze: the numerator (gross premium) is smaller in India
than in the US, and the denominator (cost hurdle) is larger. It does not say
momentum is absent. It says the **implementable** premium is thin, and that
implementation constraints — not factor selection — are what decide whether
anything survives.

### 1.2 Anomaly decay, with a genuine international nuance

McLean & Pontiff (2016) is the landmark: documented anomaly returns are ~26%
lower out-of-sample and ~58% lower after publication
([Does Academic Research Destroy Stock Return Predictability?](https://www.researchgate.net/publication/254926004_Does_Academic_Research_Destroy_Stock_Return_Predictability)).

But Jacobs & Müller
([Anomalies across the globe: Once public, no longer existent?](https://www.sciencedirect.com/science/article/abs/pii/S0304405X19301618))
find the **US is the only country with a reliable post-publication decline** in
long-short returns.

I read this cautiously in both directions. It may mean India's premia are less
arbitraged. It may also mean international evidence is noisier and less
policed, so apparent persistence is partly a data-quality artifact. Either
reading argues the same thing: **do not import a US effect size into an India
system.** Measure it here or don't claim it.

### 1.3 India-specific evidence: real, but thinner than it looks

**Fama-French in India — reasonably supported.** Multiple studies find the
three- and five-factor models outperform CAPM on Indian data, with size and
value adding genuine explanatory power
([FF5 on CNX 500, 2000-2015](https://www.inderscienceonline.com/doi/abs/10.1504/IJBG.2021.111959);
[Asset-pricing models: a case of Indian capital market](https://www.tandfonline.com/doi/full/10.1080/23322039.2020.1832732)).
One study covering 2016-23 found the **size effect outperformed the value
effect** in that window
([FF3 applicability](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5010405)).

**India momentum — evidence exists, quality is mixed.** The papers I retrieved
sit largely in lower-tier or non-indexed venues (Global Business Review, IJRAR,
ResearchGate preprints) rather than top finance journals
([Momentum Effect in Indian Stock Market: A Sectoral Study](https://journals.sagepub.com/doi/abs/10.1177/0972150915569940);
[Momentum and contrarian strategies](https://www.ijrar.org/papers/IJRAR190I031.pdf)).
Findings are directionally positive — momentum profits at 6-12 month horizons —
but this is **not** the same evidentiary weight as Jegadeesh-Titman in the US.

**This asymmetry is the single most important research conclusion for us:** the
factor with the *strongest India evidence* is value/size (fundamentals), and the
factor our system is *built on* is momentum, where India evidence is weakest.

### 1.4 Corroboration from our own backtest

Last session's walk-forward produced **DSR 0.7%** — momentum-led, after real
Indian costs, failing the 95% bar. That is not an anomalous result. It is
**exactly what the emerging-market cost literature predicts**: a thin gross
premium minus a large cost hurdle leaves nothing distinguishable from noise.

The empirical result and the literature agree. That convergence is worth more
than either alone.

---

## 2. WHAT SHOULD BE AVOIDED

**Retail technical indicators — REJECT.** Four searches surfaced no
peer-reviewed evidence that RSI, MACD, Bollinger Bands, ADX, Supertrend,
Stochastic, Aroon or Ichimoku carry *incremental* predictive information beyond
trend/momentum, in India or elsewhere. They are monotone transforms of the same
price series. The correct test is not "does it look useful" but "does it add
information the momentum factor does not already contain", and nothing I found
clears that bar.

**Candlestick and chart patterns — REJECT.** No credible evidence retrieved.
High researcher-degrees-of-freedom, subjective definitions, and a literature
dominated by practitioner material rather than peer review.

**Importing US effect sizes — REJECT.** §1.2 and §1.3.

**Probability language — REJECT** until calibrated against realised outcomes.

---

## 3. THE IDEAL SYSTEM (designed as if from scratch)

### Universe
**NSE large/mid-cap, liquidity-screened, ~150-250 names.** Not a preference —
§1.1 says the large-cap constraint is what makes an EM factor implementable at
all. Point-in-time membership required to avoid survivorship bias.

### Data
1. Adjusted daily OHLCV, ≥10 years, corporate-action correct
2. **Point-in-time fundamentals with filing dates** — the highest-value missing input
3. Index series + India VIX for regime
4. Historical index membership
5. Trading calendar discovered from data

### Feature set — the smallest defensible set
| Family | Include? | Why |
|---|---|---|
| **Value** (earnings yield, B/P) | **Yes** | Strongest India evidence (§1.3) |
| **Quality** (ROE, accruals, leverage) | **Yes** | FF5 profitability factor supported in India |
| **Momentum 12-1** | **Yes, reduced weight** | Weak India evidence, high cost sensitivity |
| **Size** | Yes, as control | Outperformed value 2016-23 |
| Low volatility | Investigate | UNVERIFIED for India |
| Technical indicators | **No** | §2 |
| Patterns | **No** | §2 |

**Four families, not twelve.** Every addition multiplies the multiple-testing
penalty; §1.2 says most of what you'd add is decaying anyway.

### Signal architecture
Cross-sectional rank → composite → liquidity/cost gate → regime scale →
top-N or NO TRADE. Costs applied **before** ranking, not after — a name whose
edge is smaller than its own cost hurdle should never reach the shortlist.

### Risk architecture
Volatility-scaled stops, position size = min(risk budget, capital, liquidity),
liquidity binding when tightest.

### Validation
Walk-forward → purged/embargoed CPCV → PBO → DSR against an honest trial count.
Frozen final test period.

### Output
Rank, evidence for, evidence against, cost hurdle, and an explicit statement
that no probability is available.

---

## 4. GAP ANALYSIS — current system vs ideal

| Area | Ideal | Current | Gap | Priority |
|---|---|---|---|---|
| **Factor mix** | Value+quality led, momentum reduced | **Momentum 50% / sector-RS 50%, quality dropped** | Built on the weakest-evidenced factor for India | **CRITICAL** |
| **Fundamentals** | PIT with filing dates | Absent; quality factor dropped at runtime | Blocks the best-evidenced factors | **CRITICAL** |
| Universe | PIT membership | Current-vintage NIFTY 200 | Survivorship bias, size unknown | HIGH |
| History | ≥10y | 990 sessions (~4y) | Insufficient for CPCV | HIGH |
| Cost gate | Before ranking | After, in Stage 7 | Cost-doomed names reach shortlist | HIGH |
| Technical indicators | None | **None** | **No gap — correct** | — |
| Probability | Never uncalibrated | Never emitted, enforced by test | **No gap — correct** | — |
| Liquidity/large-cap | Mandatory | ADTV floor + participation cap | **No gap — matches §1.1** | — |
| Look-ahead | Prevented | Verified absent, instrumented | **No gap — correct** | — |
| Validation | CPCV/PBO/DSR | Built and **executed** (DSR 0.7%) | PBO still unrun | MEDIUM |

### What to KEEP (research-aligned, verified correct)
- Liquidity gates and large-cap universe — directly supported by §1.1
- Absence of technical indicators — §2 vindicates this
- Cost model, look-ahead protections, ledger, job system, NO-TRADE discipline
- Validation machinery — it caught the negative result

### What to REMOVE
- **Sector-relative-strength at 50% weight.** It is a momentum transform; the
  redundancy check already measured ρ=0.375 against momentum. Two correlated
  momentum expressions carrying the entire signal is concentration disguised as
  diversification.

### What to MODIFY
- **Rebalance the factor mix** once fundamentals exist: value + quality primary,
  momentum secondary.
- **Move the cost hurdle before ranking.**
- **Earnings gate** — 45-session blackout rejects 85% of the universe; separate
  pre-results blackout from results-inside-hold.

### What to ADD
1. **Point-in-time fundamentals** via NSE `corporates-financial-results`
   (`filingDate` + `broadCastDate` already proven reachable — see DATA_SOURCES.md)
2. Historical index membership
3. 10 years of history
4. PBO across a real configuration sweep

---

## 5. UPSTOX vs REQUIREMENT

| Requirement | Upstox | Current code | Verdict |
|---|---|---|---|
| Daily OHLCV | Yes, from 2000 | NSE archives | Keep NSE — authoritative, free, no auth |
| Intraday | Yes, from 2022 | Not used | Not needed for EOD |
| **PIT fundamentals** | **4 annual periods, no filing date** | Absent | **Unsuitable — use NSE financial-results** |
| Historical membership | No | No | **Neither source solves it** |
| India VIX / indices | Yes | NSE archives | Keep NSE |

**Upstox does not close the critical gap.** Its fundamentals lack filing dates
and depth. The better path was already proven: NSE's `corporates-financial-results`
carries `filingDate`, which is precisely what point-in-time integrity requires.

---

## 6. PRIORITISED PLAN

**CRITICAL**
1. Ingest PIT fundamentals from NSE financial-results (filing-date gated)
2. Rebalance factors: value + quality primary, momentum secondary
3. Re-run walk-forward and DSR on the new mix

**HIGH**
4. Move cost hurdle before ranking
5. Extend history to 10 years
6. Fix the earnings gate
7. Reduce or remove sector-RS weight

**MEDIUM**
8. PBO over a configuration sweep
9. Historical membership (or document the bias quantitatively)

**LOW**
10. Low-volatility factor investigation

---

## 7. THE HONEST CONCLUSION

The engineering is sound and several design choices are **directly validated**
by the literature — large-cap liquidity screening, the absence of technical
indicators, the refusal to emit probabilities, the NO-TRADE discipline.

The **factor thesis is the problem.** The system is built primarily on
momentum, which has the *weakest* India evidence and the *highest* cost
sensitivity, while the factors with the strongest India evidence — value and
quality — are absent because the data was never ingested.

Our DSR of 0.7% is not a bug. It is the predicted outcome of implementing a
thin EM momentum premium against real Indian transaction costs.

**The fix is not more indicators or better code. It is different data.** Get
point-in-time fundamentals in, rebalance toward what the India evidence
supports, and re-run the validation. If it still fails, the honest answer is
that this approach does not work at this capital scale — and the system is now
built to tell you that rather than hide it.
