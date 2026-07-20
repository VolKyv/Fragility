"""
Builds config/universe.txt as the current S&P 500 constituent list
(= union of all Select Sector SPDR (XL*) holdings by construction).

KNOWN LIMITATION, stated for the record: current constituents backtested
to 2007 = survivorship bias. The 95% coverage filter in data_fetch will
additionally drop anything listed after ~2008, leaving roughly 330-380
names. Not fixable with free data; point-in-time constituents are paid.

Run: python -m scripts.build_universe
"""
import pandas as pd

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def build(out_path="config/universe.txt"):
    tables = pd.read_html(URL)
    df = tables[0]
    # Yahoo uses '-' where the index uses '.' (BRK.B -> BRK-B, BF.B -> BF-B)
    tickers = df["Symbol"].str.strip().str.replace(".", "-", regex=False)
    tickers = sorted(tickers.unique())
    with open(out_path, "w") as f:
        f.write("# S&P 500 constituents (= union of XL sector ETF holdings)\n")
        f.write(f"# Source: Wikipedia, retrieved at build time. N={len(tickers)}\n")
        f.write("# WARNING: current constituents -> survivorship bias in backtest.\n")
        for t in tickers:
            f.write(t + "\n")
    print(f"[build_universe] wrote {len(tickers)} tickers to {out_path}")


if __name__ == "__main__":
    build()
