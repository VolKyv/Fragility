"""
Evaluation protocol. Everything here operates on already-aligned
(signal, label) pairs and enforces two things:

1. Z-scoring is expanding (uses only data up to t), so no full-sample
   statistic leaks into an early observation's score.
2. Logit standard errors are HAC (Newey-West) with lag = N, because labels
   from overlapping forward windows are autocorrelated by construction —
   ignoring this inflates t-stats, which is exactly the failure mode the
   Harvey-Liu-Zhu bar is meant to catch.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score


def walk_forward_zscore(signal, min_periods=252):
    """Expanding z-score: at t, use mean/std of signal[:t] (excluding t)."""
    mu = signal.expanding(min_periods=min_periods).mean().shift(1)
    sd = signal.expanding(min_periods=min_periods).std().shift(1)
    z = (signal - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan).dropna()


def align(signal_z, label):
    """Inner-join signal and label on date index; returns aligned arrays."""
    df = pd.concat([signal_z.rename("signal"), label.rename("label")], axis=1, sort=True).dropna()
    return df


def evaluate_signal(signal, label, N, min_periods=252):
    """
    Runs the full pass/fail evaluation for one (signal, label) pair:
    - expanding z-score of the signal
    - AUC (walk-forward, since z-score itself never sees the future)
    - logistic regression with HAC(N) standard errors, returns t-stat on
      the signal coefficient

    Returns a dict of results. Does NOT compare to the null here — that
    comparison happens one level up in run_research.py where the null's
    AUC is available alongside every candidate's.
    """
    z = walk_forward_zscore(signal, min_periods=min_periods)
    df = align(z, label)

    if len(df) < 50 or df["label"].nunique() < 2:
        return {"n_obs": len(df), "auc": np.nan, "hac_tstat": np.nan,
                "coef": np.nan, "n_events": int(df["label"].sum()) if len(df) else 0}

    auc = roc_auc_score(df["label"], df["signal"])

    X = sm.add_constant(df["signal"])
    model = sm.Logit(df["label"], X)
    try:
        fit = model.fit(disp=0, cov_type="HAC", cov_kwds={"maxlags": N})
        coef = fit.params["signal"]
        tstat = fit.tvalues["signal"]
    except Exception as e:
        coef, tstat = np.nan, np.nan

    return {
        "n_obs": len(df),
        "n_events": int(df["label"].sum()),
        "auc": auc,
        "coef": coef,
        "hac_tstat": tstat,
    }


def event_study(signal, forward_return, pctile=80, min_periods=252):
    """
    Forward-return distribution conditional on signal being above its
    TRAILING pctile-th percentile (not full-sample — trailing, so this is
    also filtered-only). Returns mean/median forward return in the
    triggered regime vs the unconditional sample, for a plain-language
    sanity check alongside the AUC/logit numbers.
    """
    trailing_pctile = signal.expanding(min_periods=min_periods).apply(
        lambda w: np.percentile(w[:-1], pctile) if len(w) > 1 else np.nan, raw=True
    )
    triggered = signal > trailing_pctile
    df = pd.concat([triggered.rename("triggered"), forward_return.rename("fwd_ret")], axis=1, sort=True).dropna()

    if df["triggered"].sum() < 10:
        return {"n_triggered": int(df["triggered"].sum()), "mean_ret_triggered": np.nan,
                "mean_ret_baseline": np.nan}

    return {
        "n_triggered": int(df["triggered"].sum()),
        "mean_ret_triggered": df.loc[df["triggered"], "fwd_ret"].mean(),
        "median_ret_triggered": df.loc[df["triggered"], "fwd_ret"].median(),
        "mean_ret_baseline": df["fwd_ret"].mean(),
    }
