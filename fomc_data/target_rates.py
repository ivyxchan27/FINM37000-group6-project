"""
Download historical Federal Reserve target rate bounds from FRED.
"""

from io import StringIO
import pandas as pd
import requests
from fomc_data.config import START, END

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_target_series(series_id, start = START, end = END):
    """
    Download one daily federal funds target-rate series.

    Parameters:
    - series_id (str): FRED series identifier.
    - start (date-like): First date to include.
    - end (date-like): Last date to include.

    Returns (pd.DataFrame): with columns [date, series_id]
    Note: rates in percentage points
    """
    response = requests.get(FRED_URL, params = {"id": series_id}, timeout = 30)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df[series_id], errors = "coerce")
    df = df.loc[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]

    return df.reset_index(drop = True)


def fetch_target_rates(start = START, end = END):
    """
    Retrieve daily lower and upper federal funds target bounds.

    Parameters:
    - start (date-like): First date to include.
    - end (date-like): Last date to include.

    Returns (pd.DataFrame): with columns [date, target_lower, target_upper, target_midpoint]
    Note: rates in percentage points
    """
    lower = fetch_target_series("DFEDTARL", start, end)
    upper = fetch_target_series("DFEDTARU", start, end)

    df = lower.merge(upper, on = "date", how = "outer")
    df = df.rename(columns = {"DFEDTARL": "target_lower", "DFEDTARU": "target_upper"})
    df = df.sort_values("date").reset_index(drop = True)

    #use the target-range midpoint as the realized policy-rate benchmark.
    df["target_midpoint"] = (df["target_lower"] + df["target_upper"]) / 2

    return df
