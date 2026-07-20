"""
Data fetch layer. All functions return pandas objects indexed by trading date.

NOTE: written and syntax-checked in a sandboxed environment with no network
path to Yahoo Finance. Standard yfinance calls, but not exercised against
live data before this repo was handed over. Run tests/test_synthetic.py for
logic checks that don't need network; run this module directly on your own
machine first to confirm the live fetch works before trusting downstream
results.
"""
import pandas as pd
import yfinance as yf

import config


def fetch_adj_close(tickers, start=config.START_DATE):
    """Fetch adjusted close for one or more tickers. Returns a DataFrame,
    columns = tickers, even for a single ticker."""
    if isinstance(tickers, str):
        tickers = [tickers]
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"]
    else:
        px = raw[["Close"]]
        px.columns = tickers
    return px.dropna(how="all")


def fetch_ohlcv(ticker, start=config.START_DATE):
    """Fetch OHLCV for a single ticker, used for distribution-day counting
    which needs volume, not just close."""
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw[["Open", "High", "Low", "Close", "Volume"]].dropna()


def fetch_universe_returns(universe_file=config.UNIVERSE_FILE, start=config.START_DATE):
    """Daily log returns panel for the equity universe used by S1/S2/S6.
    Drops any ticker with insufficient history rather than failing outright,
    and reports what was dropped so silent gaps don't propagate."""
    tickers = [
        line.strip()
        for line in open(universe_file)
        if line.strip() and not line.strip().startswith("#")
    ]
    px = fetch_adj_close(tickers, start=start)
    coverage = px.notna().mean()
    dropped = coverage[coverage < 0.95].index.tolist()
    if dropped:
        print(f"[data_fetch] dropping {len(dropped)} tickers with <95% history coverage: {dropped}")
    px = px.drop(columns=dropped)
    px = px.dropna(how="any")  # keep only fully-overlapping dates across the surviving universe
    import numpy as np
    log_rets = np.log(px / px.shift(1)).dropna(how="all")
    return log_rets


def fetch_vix_term_structure(start=config.START_DATE):
    px = fetch_adj_close([config.VIX_TICKER, config.VIX3M_TICKER], start=start)
    px.columns = ["VIX", "VIX3M"]
    return px.dropna()


def fetch_yield_curve(start=config.START_DATE):
    px = fetch_adj_close([config.YIELD_3M_TICKER, config.YIELD_10Y_TICKER], start=start)
    px.columns = ["Y3M_x10", "Y10Y_x10"]
    # Yahoo quotes ^IRX/^TNX as yield * 10 (e.g. 45.0 == 4.50%)
    out = px / 10.0
    out.columns = ["Y3M", "Y10Y"]
    return out.dropna()


def fetch_oil(start=config.START_DATE):
    px = fetch_adj_close(config.OIL_TICKER, start=start)
    px.columns = ["CL"]
    return px.dropna()
