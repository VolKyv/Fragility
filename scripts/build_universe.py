"""
Builds config/universe.txt as the current S&P 500 constituent list
(= union of all Select Sector SPDR (XL*) holdings by construction).

KNOWN LIMITATION, stated for the record: current constituents backtested
to 2007 = survivorship bias. The 95% coverage filter in data_fetch will
additionally drop anything listed after ~2008, leaving roughly 330-380
names. Not fixable with free data; point-in-time constituents are paid.

Run: python -m scripts.build_universe
"""
from io import StringIO

import pandas as pd
import requests

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Wikipedia returns 403 to Python's default urllib User-Agent (which is
# what pd.read_html(url) uses), especially from datacenter IPs like
# GitHub Actions runners. Fetch explicitly with a browser-style UA.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}


def build(out_path="config/universe.txt"):
    resp = requests.get(URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    df = tables[0]
    # Yahoo uses '-' where the index uses '.' (BRK.B -> BRK-B, BF.B -> BF-B)
    tickers = df["Symbol"].str.strip().str.replace(".", "-", regex=False)
    tickers = sorted(tickers.unique())
    if len(tickers) < 400:
        raise RuntimeError(
            f"Only {len(tickers)} tickers parsed - page structure may have "
            "changed; refusing to write a truncated universe."
        )
    with open(out_path, "w") as f:
        f.write("# S&P 500 constituents (= union of XL sector ETF holdings)\n")
        f.write(f"# Source: Wikipedia, retrieved at build time. N={len(tickers)}\n")
        f.write("# WARNING: current constituents -> survivorship bias in backtest.\n")
        for t in tickers:
            f.write(t + "\n")
    print(f"[build_universe] wrote {len(tickers)} tickers to {out_path}")


if __name__ == "__main__":
    build()
