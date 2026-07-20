"""
End-to-end orchestrator. Computes universe-level signals ONCE (they're
market-wide fragility measures, not index-specific) and reuses them across
both SPY and QQQ label sets. Index-specific signals (distribution days) are
computed separately per index. QQQ results are written but flagged as
confirmatory, not independent headline tests — see README.

Run: python -m src.run_research
"""
import numpy as np
import pandas as pd

import config
from src import data_fetch, signals, labels, backtest


def build_signal_set(universe_rets, vix_df, yield_df, oil_df):
    """Universe/macro signals, shared across both indices."""
    out = {}
    out["absorption_ratio"] = signals.absorption_ratio(universe_rets)
    out["turbulence"] = signals.turbulence_index(universe_rets)
    out["avg_pairwise_corr"] = signals.avg_pairwise_correlation(universe_rets)  # S0 null
    out["vix_term"] = signals.vix_term_structure(vix_df)
    slope, delta = signals.yield_curve_slope(yield_df)
    out["yield_slope"] = slope
    out["yield_slope_delta"] = delta
    oil_ret, oil_vol = signals.oil_proxy(oil_df)
    out["oil_ret_20d"] = oil_ret
    out["oil_vol_20d"] = oil_vol
    return out


def forward_return(price, N):
    return (price.shift(-N) / price - 1).rename(f"fwd_ret_{N}d")


def run():
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    print("[run_research] fetching universe returns...")
    universe_rets = data_fetch.fetch_universe_returns()

    print("[run_research] fetching VIX term structure, yield curve, oil...")
    vix_df = data_fetch.fetch_vix_term_structure()
    yield_df = data_fetch.fetch_yield_curve()
    oil_df = data_fetch.fetch_oil()

    shared_signals = build_signal_set(universe_rets, vix_df, yield_df, oil_df)

    all_rows = []
    robustness_rows = []

    for index_ticker in config.INDICES:
        print(f"[run_research] processing {index_ticker}...")
        ohlcv = data_fetch.fetch_ohlcv(index_ticker)
        price = ohlcv["Close"]
        dd_count = signals.distribution_days(ohlcv)

        index_signals = dict(shared_signals)
        index_signals["dd_count"] = dd_count

        label_sets = labels.build_all_labels(price, config.TRACKS, config.ROBUSTNESS_TRACK)

        for track_key, (label, drawdown) in label_sets.items():
            is_robustness = track_key in config.ROBUSTNESS_TRACK
            spec = (config.ROBUSTNESS_TRACK if is_robustness else config.TRACKS)[track_key]
            N = spec["N"]
            fwd_ret = forward_return(price, N)

            for sig_name, sig_series in index_signals.items():
                is_headline = (not is_robustness) and (index_ticker == "SPY") and \
                              ((track_key, sig_name) in config.HEADLINE_CELLS)

                result = backtest.evaluate_signal(sig_series, label, N)
                event = backtest.event_study(sig_series, fwd_ret)

                row = {
                    "index": index_ticker,
                    "track": track_key,
                    "track_label": spec["label"],
                    "signal": sig_name,
                    "is_headline": is_headline,
                    "is_confirmatory_only": index_ticker != "SPY" and not is_robustness,
                    **result,
                    **{f"event_{k}": v for k, v in event.items()},
                }
                (robustness_rows if is_robustness else all_rows).append(row)

    results = pd.DataFrame(all_rows).sort_values(
        ["track", "index", "is_headline"], ascending=[True, True, False]
    )
    robustness = pd.DataFrame(robustness_rows)

    # Null comparison: does each signal beat avg_pairwise_corr's AUC on the same cell?
    null_auc = results[results["signal"] == "avg_pairwise_corr"].set_index(
        ["index", "track"]
    )["auc"]
    results["null_auc"] = results.apply(
        lambda r: null_auc.get((r["index"], r["track"]), np.nan), axis=1
    )
    results["beats_null"] = results["auc"] > results["null_auc"]
    results["passes_hlz"] = results["hac_tstat"].abs() > config.HLZ_T_STAT_BAR
    results["survives"] = results["beats_null"] & results["passes_hlz"]

    out_path = config.OUTPUT_DIR / "results.csv"
    results.to_csv(out_path, index=False)
    print(f"[run_research] wrote {out_path} ({len(results)} rows)")

    if len(robustness):
        rob_path = config.OUTPUT_DIR / "results_robustness.csv"
        robustness.to_csv(rob_path, index=False)
        print(f"[run_research] wrote {rob_path} ({len(robustness)} rows)")

    headline = results[results["is_headline"]]
    if len(headline):
        print("\n[run_research] HEADLINE CELLS (pre-registered):")
        print(headline[["track_label", "signal", "auc", "hac_tstat", "survives"]].to_string(index=False))

    return results, robustness


if __name__ == "__main__":
    run()
