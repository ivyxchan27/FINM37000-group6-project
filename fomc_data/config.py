"""
Configuration for the FOMC meeting and realized rate data pipeline.
"""

from pathlib import Path
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent

#sample period
START = pd.Timestamp("2022-01-01")
END = pd.Timestamp("2026-09-30")

#data directories
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

#processed data inputs and outputs.
EFFR_CSV = PROCESSED_DIR / "effr_daily.csv"
SOFR_CSV = PROCESSED_DIR / "sofr_daily.csv"
FOMC_CSV = PROCESSED_DIR / "fomc_meetings.csv"
SOFR_FUTURES_PARQUET = PROCESSED_DIR / "sofr_futures_settlements.parquet"
FOMC_ALIGNMENT_CSV = PROCESSED_DIR / "fomc_futures_alignment.csv"


def ensure_dirs():
    """
    Create the raw and processed data directories if needed.

    Returns: None.
    """
    for path in (RAW_DIR, PROCESSED_DIR):
        path.mkdir(parents = True, exist_ok = True)
