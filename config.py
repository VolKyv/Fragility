"""
Central configuration. Change parameters here, not inline in analysis code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Labels: pre-registered before any signal testing ---
TRACKS = {
    "A": {"D": -0.08, "N": 40, "label": "intermediate (-8%/40d)"},
    "B": {"D": -0.05, "N": 20, "label": "short (-5%/20d)"},
}
ROBUSTNESS_TRACK = {"C": {"D": -0.10, "N": 60, "label": "robustness (-10%/60d)"}}

# Pre-registered headline cells: (track, signal). Everything else exploratory.
HEADLINE_CELLS = [("A", "turbulence"), ("B", "vix_term")]

HLZ_T_STAT_BAR = 3.0

# --- Universe / indices ---
INDICES = ["SPY", "QQQ"]
UNIVERSE_FILE = ROOT / "config" / "universe.txt"

# --- External tickers ---
VIX_TICKER = "^VIX"
VIX3M_TICKER = "^VIX3M"
YIELD_3M_TICKER = "^IRX"   # quoted as yield * 10 on Yahoo
YIELD_10Y_TICKER = "^TNX"  # quoted as yield * 10 on Yahoo
OIL_TICKER = "CL=F"        # front-month WTI, not the curve

START_DATE = "2007-01-01"

# --- Signal computation params ---
MIN_HISTORY_DAYS = 252          # min lookback before scoring any signal
DD_WINDOW = 25                  # O'Neil distribution-day rolling window
DD_DECLINE_THRESHOLD = -0.002   # -0.2% close-to-close counts as a down day
DD_CLEAR_RALLY = 0.05           # 5% rally from cluster low clears the count
AR_RMT_CLEAN = True             # Marchenko-Pastur eigenvalue cleaning for S1
AR_FALLBACK_FRACTION = 0.2      # used only if AR_RMT_CLEAN finds 0 signal eigenvalues

OUTPUT_DIR = ROOT / "output"
