"""
Historical FOMC policy announcement dates.

Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
"""

import pandas as pd
from fomc_data.config import START, END


#last days of regularly scheduled meetings, when decisions were announced.
FOMC_DATES = {2022: ["01-26", "03-16", "05-04", "06-15", "07-27", "09-21", "11-02", "12-14"],
              2023: ["02-01", "03-22", "05-03", "06-14", "07-26", "09-20", "11-01", "12-13"],
              2024: ["01-31", "03-20", "05-01", "06-12", "07-31", "09-18", "11-07", "12-18"],
              2025: ["01-29", "03-19", "05-07", "06-18", "07-30", "09-17", "10-29", "12-10"],
              2026: ["01-28", "03-18", "04-29", "06-17", "07-29", "09-16"]}


def fetch_fomc_calendar(start = START, end = END):
    """
    Returns historical FOMC announcement dates.

    Parameters:
    - start (date-like): First date to include.
    - end (date-like): Last date to include.

    Returns (pd.DataFrame): with columns [meeting_date, year, meeting_number]
    """
    dates = [f"{year}-{month_day}" for year, meeting_dates in FOMC_DATES.items() for month_day in meeting_dates]

    df = pd.DataFrame({"meeting_date": pd.to_datetime(dates)})

    df = df.loc[(df["meeting_date"] >= pd.Timestamp(start)) & (df["meeting_date"] <= pd.Timestamp(end))].copy()

    df = df.sort_values("meeting_date").reset_index(drop=True)

    #number meetings sequentially within each calendar year.
    df["year"] = df["meeting_date"].dt.year
    df["meeting_number"] = df.groupby("year").cumcount() + 1

    return df[["meeting_date", "year", "meeting_number"]]
