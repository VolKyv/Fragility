"""
Tests against synthetic data only — no network required. These validate
that the pipeline runs end-to-end and that shapes/types are correct. They
do NOT validate that any signal actually predicts anything; that's what
run_research.py against real data is for.
"""
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import signals, labels, backtest


@pytest.fixture
def synthetic_universe():
    rng = np.random.default_rng(42)
    n_days, n_assets = 500, 15
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rets = rng.normal(0, 0.01, size=(n_days, n_assets))
    return pd.DataFrame(rets, index=dates, columns=[f"A{i}" for i in range(n_assets)])


@pytest.fixture
def synthetic_price():
    rng = np.random.default_rng(7)
    n_days = 500
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rets = rng.normal(0.0003, 0.012, size=n_days)
    price = 100 * np.exp(np.cumsum(rets))
    return pd.Series(price, index=dates, name="Close")


@pytest.fixture
def synthetic_ohlcv(synthetic_price):
    rng = np.random.default_rng(3)
    vol = rng.integers(1_000_000, 5_000_000, size=len(synthetic_price))
    return pd.DataFrame({
        "Open": synthetic_price.values,
        "High": synthetic_price.values * 1.005,
        "Low": synthetic_price.values * 0.995,
        "Close": synthetic_price.values,
        "Volume": vol,
    }, index=synthetic_price.index)


def test_absorption_ratio_shape_and_bounds(synthetic_universe):
    ar = signals.absorption_ratio(synthetic_universe, cov_window=100)
    assert len(ar) > 0
    assert (ar >= 0).all() and (ar <= 1.0001).all()


def test_turbulence_nonnegative(synthetic_universe):
    turb = signals.turbulence_index(synthetic_universe, cov_window=100)
    assert len(turb) > 0
    assert (turb >= 0).all()


def test_avg_pairwise_corr_bounds(synthetic_universe):
    corr = signals.avg_pairwise_correlation(synthetic_universe, cov_window=100)
    assert len(corr) > 0
    assert (corr >= -1.0001).all() and (corr <= 1.0001).all()


def test_forward_drawdown_labels(synthetic_price):
    label, drawdown = labels.forward_drawdown_labels(synthetic_price, D=-0.05, N=20)
    assert len(label) == len(synthetic_price) - 20
    assert set(label.unique()).issubset({0.0, 1.0})
    # NOTE: forward drawdown is NOT guaranteed <= 0 here. It's
    # (min(future window) - price[t]) / price[t]; under positive drift the
    # future window can stay entirely above price[t]. Only check the label
    # logic is internally consistent with the drawdown values.
    assert (label[drawdown <= -0.05] == 1.0).all()
    assert (label[drawdown > -0.05] == 0.0).all()


def test_distribution_days_bounds(synthetic_ohlcv):
    dd = signals.distribution_days(synthetic_ohlcv, window=25)
    dd = dd.dropna()  # leading rows before `window` history accrues are NaN by design
    assert len(dd) > 0
    assert (dd >= 0).all() and (dd <= 25).all()


def test_walkforward_zscore_no_lookahead(synthetic_universe):
    turb = signals.turbulence_index(synthetic_universe, cov_window=100)
    z = backtest.walk_forward_zscore(turb, min_periods=50)
    # z-score at the first valid point must not equal a full-sample z-score
    # (sanity check that it's genuinely expanding, not full-sample)
    full_sample_z = (turb - turb.mean()) / turb.std()
    common = z.index.intersection(full_sample_z.index)[:5]
    assert not np.allclose(z.loc[common], full_sample_z.loc[common])


def test_evaluate_signal_runs(synthetic_universe, synthetic_price):
    turb = signals.turbulence_index(synthetic_universe, cov_window=100)
    label, _ = labels.forward_drawdown_labels(synthetic_price, D=-0.05, N=20)
    result = backtest.evaluate_signal(turb, label, N=20, min_periods=100)
    assert "auc" in result and "hac_tstat" in result


def test_event_study_runs(synthetic_universe, synthetic_price):
    turb = signals.turbulence_index(synthetic_universe, cov_window=100)
    fwd_ret = (synthetic_price.shift(-20) / synthetic_price - 1)
    result = backtest.event_study(turb, fwd_ret, min_periods=100)
    assert "n_triggered" in result
