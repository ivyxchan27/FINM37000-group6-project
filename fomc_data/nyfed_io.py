"""
Retrieve daily Effective Federal Funds Rate data from the New York Fed.
"""

import pandas as pd
import requests
from fomc_data.config import START, END


NYFED_EFFR_URL = ("https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json")


def fetch_effr(start = START, end = END):
    """
    Fetch daily EFFR observations from the New York Fed.

    Parameters:
    - start (date-like): First observation date to request.
    - end (date-like): Last observation date to request.

    Returns (pd.DataFrame): with columns [date, effr]
    Note: EFFR is in percentage points
    """
    params = {"startDate": pd.Timestamp(start).strftime("%Y-%m-%d"),
              "endDate": pd.Timestamp(end).strftime("%Y-%m-%d")}

    response = requests.get(NYFED_EFFR_URL, params = params, timeout = 30)
    response.raise_for_status()

    payload = response.json()

    observations = payload["refRates"]
    df = pd.DataFrame(observations)
    df = df.rename(columns = {"effectiveDate": "date", "percentRate": "effr"})
    df = df[["date", "effr"]].copy()

    #normalize API fields for downstream time-series analysis.
    df["date"] = pd.to_datetime(df["date"])
    df["effr"] = pd.to_numeric(df["effr"], errors = "coerce")

    return df.sort_values("date").reset_index(drop = True)
