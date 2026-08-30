# v3 search code

Run against a staged copy of the curated store; `data.py` holds the path. These
are the scripts as they ran, not a tidied library — they are here so a number in
`research/V3_SEARCH.md` can be traced to the code that produced it.

Order:

1. `pit_fund.py` — point-in-time fundamentals: real filing dates where they
   exist, measured p99 disclosure lag where they do not.
2. `themes.py` — 93 factors in 8 themes, with the theme map.
3. `panel2.py` — the themed panel, 209,048 × 115.
4. `seal2.py` — seals windows A and B **before anything is fitted**.
5. `guard.py` — the forward-looking column guard. `mae5`/`mfe5` leaked into an
   early screen at IC 0.22–0.25; factor columns now come from declared themes
   only.
6. `screen2.py` — the placebo-alignment screen.
7. `stability.py` — both halves of each factor's own life.
8. `redundancy.py` / `survivors.py` — the correlation matrix and the
   order-independent survivor pass.
9. `composite.py` — `cap_weights`: cap, floor, coverage cap.
10. `search3.py`, `run_s3b/c/d/e.py` — level-1 methods, level-2 weights, the
    floor, the book.
11. `lowturn.py`, `entryonly.py` — the cost curve and the entry-only floor.
12. `freeze2.py` — writes the frozen configuration and its sha256.
13. `holdout3.py` / `holdout3b.py` — one blind evaluation per window. `holdout3b`
    selects `FROZEN_V3_PRE_B.json` for window B, because the main frozen config
    was fitted through 2024-10 and is in-sample for B.
14. `prb.py` — the permuted-label power test.

The shipped scorer is NOT this code. It is `src/prosignal/features/v3.py` and
`v3_factors.py`, which recompute the same 22 factors in the production store's
own terms; `tests/test_v3_score.py` pins them against these definitions.
