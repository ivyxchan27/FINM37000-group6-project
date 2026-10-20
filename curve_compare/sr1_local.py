"""Local SR1 averaging helper for the synthetic calendar-day SOFR series.

`sr1_average_rate` below is a small, deliberately minimal reimplementation of
the same SR1 convention used by `sofr_pricing.sr1_rate`: the day-weighted
average of daily SOFR over a calendar month, with every calendar day counted once.

It is simpler than `sofr_pricing.sr1_rate` because the synthetic daily SOFR
series built in `synthetic.py` already has exactly one value per CALENDAR
day (it skips the business-day fixing calendar entirely, see `synthetic.py`),
so there is no weekend/holiday carry-forward to handle here.

Use `sofr_pricing.sr1_rate` when the data pipeline supplies a business-day
fixing path; the signature (daily rate Series, start date, end date) is
deliberately the same.
"""

import pandas as pd


def sr1_average_rate(daily_sofr: pd.Series, start, end) -> float:
    """Day-weighted average of `daily_sofr` (in percent) over [start, end], inclusive."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    window = daily_sofr.loc[start:end]
    expected_days = (end - start).days + 1
    if len(window) != expected_days:
        raise ValueError(
            f"expected {expected_days} daily SOFR fixings between "
            f"{start.date()} and {end.date()}, found {len(window)}"
        )
    return float(window.mean())
