"""
Label construction: forward drawdown over a fixed horizon N, thresholded
at D. Labels are inherently forward-looking BY DESIGN (that's what makes
them something to predict) — the leakage risk is on the SIGNAL side, not
here. See backtest.py for how signals are prevented from seeing t+1..t+N.
"""
import numpy as np
import pandas as pd


def forward_drawdown_labels(price, D, N):
    """
    For each date t, drawdown = (min(price[t+1 : t+N+1]) - price[t]) / price[t].
    Label = 1 if drawdown <= D (D is negative, e.g. -0.08).

    The last N rows of the series cannot be labeled (no forward window) and
    are dropped, not filled — silently filling them would bias any AUC
    computed near the end of the sample.
    """
    n = len(price)
    values = price.values
    dd = np.full(n, np.nan)

    for t in range(n - N):
        fwd_min = values[t + 1: t + N + 1].min()
        dd[t] = (fwd_min - values[t]) / values[t]

    drawdown = pd.Series(dd, index=price.index, name="fwd_drawdown")
    label = (drawdown <= D).astype(float)
    label[drawdown.isna()] = np.nan
    label.name = "label"
    return label.dropna(), drawdown.dropna()


def build_all_labels(price, tracks, robustness_track=None):
    """Returns dict {track_key: (label_series, drawdown_series)} for both
    primary tracks and, if provided, the robustness track."""
    all_tracks = dict(tracks)
    if robustness_track:
        all_tracks.update(robustness_track)

    out = {}
    for key, spec in all_tracks.items():
        label, drawdown = forward_drawdown_labels(price, spec["D"], spec["N"])
        out[key] = (label, drawdown)
    return out
