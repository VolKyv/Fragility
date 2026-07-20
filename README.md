# fragility-signals

Research repo for correction-risk signal testing. Companion to the `regime`
(jumpmodels/HMM) and `rrg-dash` (RRG + Distribution Monitor) Vercel projects.
This repo is the research/backtest layer only — nothing here is deployed.
Production scoring is meant to be ported into `regime`'s scheduled job once a
signal survives this pipeline, writing a JSON snapshot that `rrg-dash` reads.

## Spec (pre-registered before any results were run)

**Labels — forward drawdown, two tracks tested in parallel + one robustness
track:**

| Track | D (threshold) | N (sessions) | Role |
|---|---|---|---|
| A | -8% | 40 | intermediate — feeds Layer 1 exposure signal |
| B | -5% | 20 | short-fuse — feeds stop/no-new-buys signal |
| C | -10% | 60 | robustness appendix only, not a primary test |

Computed on **both SPY and QQQ**. SPY is the primary label set. QQQ is
**confirmatory replication, not an independent discovery run** — running full
headline tests separately on QQQ would double the multiple-testing surface
that the Harvey-Liu-Zhu bar is meant to guard against. A signal is credible
if it passes on SPY and shows the same sign/magnitude on QQQ.

**Signals under test:**

| Code | Signal | Notes |
|---|---|---|
| S1 | RMT-cleaned Absorption Ratio | Marchenko-Pastur eigenvalue cleaning. This is a **fresh implementation** written for this repo — reconcile against your existing pure-NumPy AR before treating results as final; it was not ported from that code. |
| S2 | Turbulence Index (Kritzman & Li 2010) | Mahalanobis distance vs expanding-window mean/covariance |
| S3 | VIX/VIX3M term structure ratio | backwardation = stress |
| S4 | Yield curve: 10y-3m slope + 20d Δslope | horizon mismatch flagged — literature supports recession timing (12-18mo), not this timeframe. Included as falsifiable candidate only. |
| S5 | Oil proxy: 20d return + realized vol of front-month CL=F | proxy for the true futures curve; true curve needs contract-roll data not implemented here |
| S6 | O'Neil distribution-day count | 25-session window, 5% clearing rule. Independent Python port for research parity with `api/distribution.js` — may not match it exactly, diff before treating as ground truth. |
| S0 (null) | Average pairwise correlation | every signal above must beat this out-of-sample or it's dropped |

**Pre-registered headline cells** (everything else is exploratory):
- Track A × S2 (Turbulence)
- Track B × S3 (VIX term structure)

**Protocol:** expanding-window, filtered-only (no data from t or later used to
score t). Z-score standardization is itself expanding. AUC + logistic
regression with HAC (Newey-West) standard errors, lag = N. Pass bar: HAC
t-stat > 3.0 (Harvey-Liu-Zhu), AND beats the null signal's AUC on the same
label.

## Known limitations, stated plainly

- **Live data fetch is untested in the environment that built this repo** —
  the sandbox used to write this code has no network path to Yahoo Finance,
  so `data_fetch.py` is correct-by-construction (standard yfinance calls) but
  has only been exercised against synthetic data (`tests/test_synthetic.py`).
  Run the tests with real data on your machine before trusting output.
- `universe.txt` ships with a placeholder ~40-name large-cap list. Replace it
  with your actual ~75-stock universe before running S1/S2/S6 for real —
  results with the placeholder list are not meaningful for your framework.
- Sample size is the binding constraint regardless of methodology: 2007-present
  gives roughly 8-12 Track-A-scale correction events. Confidence intervals on
  any single signal will be wide. This is stated, not fixable.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m src.run_research
```

Writes `output/results.csv` (one row per index × track × signal) and
`output/results_robustness.csv` (Track C).

## Test (no network required)

```bash
python -m pytest tests/ -v
```
