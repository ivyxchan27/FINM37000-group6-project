"""IMM dates and reference periods for SR1 and SR3 contracts.

These mirror `third_wednesday` and `reference_period` in `sofr_data/contracts.py`
on the `databento-setup` branch (PR #16). They are here so that #4 does not
depend on an unmerged PR (as promised in the PR #12 review). Once PR #16 is
merged, this module can be replaced by an import from `sofr_data.contracts`.

Conventions (CME):
- SR1: the reference period is the whole calendar month.
- SR3: the reference period runs from the third Wednesday of the contract month
  up to, but NOT including, the third Wednesday three months later.
Both periods are returned with an INCLUSIVE end date.
"""

import pandas as pd

IMM_MONTHS = (3, 6, 9, 12)


def third_wednesday(year, month):
    """Return the third Wednesday of a month (an IMM date for Mar/Jun/Sep/Dec).

    Example:
        >>> third_wednesday(2024, 12).date().isoformat()
        '2024-12-18'
        >>> third_wednesday(2025, 3).date().isoformat()
        '2025-03-19'
    """
    first_day = pd.Timestamp(year=year, month=month, day=1)
    # Monday is 0, ..., Wednesday is 2.
    days_to_first_wednesday = (2 - first_day.dayofweek) % 7
    return first_day + pd.Timedelta(days=days_to_first_wednesday + 14)


def sr1_reference_period(year, month):
    """First and last day (inclusive) of an SR1 contract month."""
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(0)
    return start, end


def sr3_reference_period(year, month):
    """First and last day (inclusive) of an SR3 reference quarter.

    Example:
        >>> [d.date().isoformat() for d in sr3_reference_period(2024, 12)]
        ['2024-12-18', '2025-03-18']
    """
    if month not in IMM_MONTHS:
        raise ValueError(f"SR3 contract months are {IMM_MONTHS}, got {month}")
    start = third_wednesday(year, month)
    three_months_later = pd.Timestamp(year=year, month=month, day=1) + pd.DateOffset(months=3)
    end = third_wednesday(three_months_later.year, three_months_later.month) - pd.Timedelta(days=1)
    return start, end


def contract_strip(obs_date, n_sr1=13, n_sr3=6):
    """Build a table of SR1 and SR3 contracts alive on `obs_date`.

    SR1: the contract for the month of `obs_date` and the next months.
    SR3: every quarterly contract whose reference period has not ended yet,
    starting with the one in progress.

    Returns:
        A DataFrame with columns contract_id, root, reference_start, reference_end,
        the same column names as the data panel from #1.
    """
    obs_date = pd.Timestamp(obs_date)
    rows = []

    # SR1: this month and the following months.
    month_start = pd.Timestamp(year=obs_date.year, month=obs_date.month, day=1)
    for i in range(n_sr1):
        first = month_start + pd.DateOffset(months=i)
        start, end = sr1_reference_period(first.year, first.month)
        rows.append({"contract_id": f"SR1-{first.year}-{first.month:02d}", "root": "SR1",
                     "reference_start": start, "reference_end": end})

    # SR3: start a year back and keep quarters that have not finished yet.
    first_quarter = pd.Timestamp(year=obs_date.year - 1, month=12, day=1)
    count = 0
    i = 0
    while count < n_sr3:
        first = first_quarter + pd.DateOffset(months=3 * i)
        start, end = sr3_reference_period(first.year, first.month)
        if end >= obs_date:
            rows.append({"contract_id": f"SR3-{first.year}-{first.month:02d}", "root": "SR3",
                         "reference_start": start, "reference_end": end})
            count = count + 1
        i = i + 1

    return pd.DataFrame(rows)
