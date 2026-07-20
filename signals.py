"""
Signal computation. Every signal here is a pandas Series indexed by date,
scored using ONLY information available up to and including that date
(filtered, not smoothed) — this is the same constraint applied to the HMM
regime work, and it's what makes the walk-forward backtest in backtest.py
honest.

Where a signal needs an estimation window (AR, Turbulence, null correlation),
the window is expanding (all history up to t), not the full sample.
"""
import numpy as np
import pandas as pd

import config


def _expanding_windows(n_rows, min_history):
    """Yield (t, window_start) pairs: score index t using rows
    [window_start, t) — strictly excludes t itself, i.e. current-day info
    is never used to score the current day's fragility/turbulence."""
    for t in range(min_history, n_rows):
        yield t, 0  # expanding: window always starts at 0


def absorption_ratio(returns_panel, rmt_clean=config.AR_RMT_CLEAN,
                      fallback_fraction=config.AR_FALLBACK_FRACTION,
                      min_history=config.MIN_HISTORY_DAYS):
    """
    S1. Kritzman et al. (2011) Absorption Ratio: fraction of total variance
    explained by the top-k eigenvalues of the return covariance matrix.
    k is chosen by Marchenko-Pastur eigenvalue cleaning (RMT) when
    rmt_clean=True: only eigenvalues exceeding the theoretical noise
    upper bound are counted as "signal" factors. Falls back to a fixed
    fraction of assets if MP cleaning finds zero signal eigenvalues
    (can happen in short/quiet windows).

    Uses an EXPANDING window (all history to date), consistent with
    filtered-only scoring elsewhere in this repo.
    """
    values = returns_panel.values
    n_rows, n_assets = values.shape
    idx = returns_panel.index
    out = pd.Series(index=idx, dtype=float)

    for t in range(min_history, n_rows):
        window = values[:t]  # strictly before t
        T = window.shape[0]
        cov = np.cov(window, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)[::-1]
        eigvals = np.clip(eigvals, 0, None)
        total = eigvals.sum()
        if total <= 0:
            continue

        if rmt_clean:
            q = n_assets / T
            sigma2 = eigvals.mean()
            lambda_plus = sigma2 * (1 + np.sqrt(q)) ** 2
            n_components = int((eigvals > lambda_plus).sum())
            if n_components == 0:
                n_components = max(1, int(np.ceil(fallback_fraction * n_assets)))
        else:
            n_components = max(1, int(np.ceil(fallback_fraction * n_assets)))

        out.iloc[t] = eigvals[:n_components].sum() / total

    return out.dropna()


def turbulence_index(returns_panel, min_history=config.MIN_HISTORY_DAYS):
    """
    S2. Kritzman & Li (2010) Turbulence: Mahalanobis distance of day t's
    return vector from the expanding-window mean/covariance estimated on
    data strictly before t. Uses pseudo-inverse for numerical stability
    when the universe is large relative to history.
    """
    values = returns_panel.values
    n_rows, n_assets = values.shape
    idx = returns_panel.index
    out = pd.Series(index=idx, dtype=float)

    for t in range(min_history, n_rows):
        window = values[:t]
        mu = window.mean(axis=0)
        cov = np.cov(window, rowvar=False)
        cov_inv = np.linalg.pinv(cov)
        diff = values[t] - mu
        out.iloc[t] = float(diff @ cov_inv @ diff.T)

    return out.dropna()


def avg_pairwise_correlation(returns_panel, min_history=config.MIN_HISTORY_DAYS):
    """
    S0 (null). Average pairwise correlation across the universe, expanding
    window. Every other signal must beat this out-of-sample or it's dropped
    — this is the hardwired bar carried over from the existing AR work.
    """
    values = returns_panel.values
    n_rows, n_assets = values.shape
    idx = returns_panel.index
    out = pd.Series(index=idx, dtype=float)
    iu = np.triu_indices(n_assets, k=1)

    for t in range(min_history, n_rows):
        window = values[:t]
        corr = np.corrcoef(window, rowvar=False)
        out.iloc[t] = np.nanmean(corr[iu])

    return out.dropna()


def vix_term_structure(vix_df):
    """S3. VIX/VIX3M ratio. >1 = backwardation = stress. No lookahead risk
    (contemporaneous, published level, not an estimated statistic)."""
    return (vix_df["VIX"] / vix_df["VIX3M"]).rename("vix_term")


def yield_curve_slope(yield_df, delta_window=20):
    """S4. 10y-3m slope, in percentage points, and its 20-day change.
    Flagged in the spec as a horizon mismatch (recession-timing literature,
    not correction-timing) — included as a falsifiable candidate."""
    slope = (yield_df["Y10Y"] - yield_df["Y3M"]).rename("yield_slope")
    delta = slope.diff(delta_window).rename("yield_slope_delta")
    return slope, delta


def oil_proxy(oil_df, ret_window=20, vol_window=20):
    """S5. Proxy for the oil futures curve using front-month spot only:
    20d return (captures shocks) and 20d realized vol (captures regime
    change in the commodity). Not the true curve — see README."""
    logret = np.log(oil_df["CL"] / oil_df["CL"].shift(1))
    ret_20d = oil_df["CL"].pct_change(ret_window).rename("oil_ret_20d")
    vol_20d = (logret.rolling(vol_window).std() * np.sqrt(252)).rename("oil_vol_20d")
    return ret_20d, vol_20d


def distribution_days(ohlcv, window=config.DD_WINDOW,
                       decline_threshold=config.DD_DECLINE_THRESHOLD,
                       clear_rally=config.DD_CLEAR_RALLY):
    """
    S6. O'Neil-spec distribution day count. A day counts as distribution if
    close-to-close return <= decline_threshold on volume higher than the
    prior day. Rolling `window`-session count, cleared if price rallies
    >= clear_rally from the low of the current DD cluster.

    This is an independent Python port for research parity with the
    existing api/distribution.js endpoint — diff outputs before treating
    this as ground truth; the two were not built from shared code.
    """
    close = ohlcv["Close"]
    vol = ohlcv["Volume"]
    ret = close.pct_change()
    is_dd = (ret <= decline_threshold) & (vol > vol.shift(1))

    dd_count = pd.Series(index=close.index, dtype=float)
    cluster_low = None
    for i in range(window, len(close)):
        window_slice = is_dd.iloc[i - window + 1: i + 1]
        count = int(window_slice.sum())

        # 5% clearing rule: rally from the lowest close since the most
        # recent distribution day inside the window clears the count.
        dd_dates_in_window = window_slice[window_slice].index
        if len(dd_dates_in_window) > 0:
            low_since_last_dd = close.loc[dd_dates_in_window[-1]: close.index[i]].min()
            rally = (close.iloc[i] / low_since_last_dd) - 1
            if rally >= clear_rally:
                count = 0

        dd_count.iloc[i] = count

    return dd_count.rename("dd_count")
