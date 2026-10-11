"""Theoretical SR1 and SR3 futures prices from a path of SOFR fixings.

Units
-----
Rates are in PERCENT (4.25 means 4.25%).
Prices are index points (95.75 means an implied rate of 4.25%).
The link between them is always:  price = 100 - rate

Input format
------------
A "fixing path" is a pandas Series:
  - the index holds the fixing dates (business days),
  - the values hold the SOFR rate for that date, in percent.
A fixing applies to every calendar day from its own date until the next
fixing date. So a Friday fixing also covers Saturday and Sunday.

The two contracts
-----------------
SR1 (One-Month SOFR) settles on the AVERAGE of the daily SOFR rate over a
calendar month. Each calendar day counts once:

    rate = (sum of daily rates) / (number of days)

SR3 (Three-Month SOFR) settles on SOFR COMPOUNDED over a reference quarter
(from one IMM date to the next), using an actual/360 day count:

    A    = product over fixings of (1 + days_k / 360 * rate_k / 100)
    rate = (A - 1) * 360 / total_days * 100

where days_k is the number of calendar days that fixing k is in force inside
the reference period.

Reference periods include both the start day and the end day. For SR3 the end
day is the day BEFORE the closing IMM date.

Not implemented yet: CME's final-settlement price rounding. Check the
contract specification before adding it.
"""

import pandas as pd

from sofr_pricing.fixing_calendar import MAX_CARRY_DAYS

# SOFR uses an actual/360 day count: a year is treated as 360 days.
DAY_COUNT_BASIS = 360


def check_fixings(fixings):
    """Stop early, with a clear message, if the fixings look wrong."""
    if not isinstance(fixings.index, pd.DatetimeIndex):
        raise TypeError("fixings must be indexed by a DatetimeIndex of fixing dates")
    if len(fixings) == 0:
        raise ValueError("fixings is empty")
    if fixings.isna().any():
        raise ValueError("fixings contains missing values")
    if not (fixings.index.is_monotonic_increasing and fixings.index.is_unique):
        raise ValueError("fixing dates must be sorted and unique")


def get_rate_blocks(fixings, start, end):
    """Cut the period [start, end] into blocks of days that share one fixing.

    Walks through the period one calendar day at a time and records, for each
    day, which fixing is in force. Consecutive days that use the same fixing
    are grouped into one block.

    Returns:
        A list of [rate, number_of_days] pairs, in date order.
        Example for a week with fixings on Wed, Thu, Fri, Mon, Tue:
        [[rate_wed, 1], [rate_thu, 1], [rate_fri, 3], [rate_mon, 1], [rate_tue, 1]]
        (Friday has 3 days because it also covers Saturday and Sunday.)

    Raises:
        ValueError: if the fixings do not properly cover the period.
    """
    check_fixings(fixings)
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    if end < start:
        raise ValueError(f"period end {end.date()} is before start {start.date()}")

    # Plain Python lists are easier to read than pandas objects.
    dates = list(fixings.index)
    rates = [float(value) for value in fixings.values]

    # The first fixing must be on or before the start day. (If the period starts 
    # on a Saturday, its rate comes from the Friday before.)
    if dates[0] > start:
        raise ValueError(
            f"fixings start {dates[0].date()}, after the period start "
            f"{start.date()}; include the fixing on or before the start"
        )

    # The last fixing must be close enough to the end day that carrying it
    # forward is reasonable (a long weekend at most).
    if (end - dates[-1]).days > MAX_CARRY_DAYS - 1:
        raise ValueError(
            f"fixings end {dates[-1].date()}, too early to cover the period "
            f"end {end.date()}"
        )

    # Step 1: find the fixing in force on the first day, which is the latest
    # fixing dated on or before `start`.
    position = 0
    while position + 1 < len(dates) and dates[position + 1] <= start:
        position = position + 1

    # Step 2: walk through the period one calendar day at a time.
    blocks = []
    last_position = None  # which fixing the previous day used
    day = start
    while day <= end:
        # If a newer fixing has started by today, switch to it.
        while position + 1 < len(dates) and dates[position + 1] <= day:
            gap_in_days = (dates[position + 1] - dates[position]).days
            if gap_in_days > MAX_CARRY_DAYS:
                raise ValueError(
                    "gap between consecutive fixings is longer than a long weekend"
                )
            position = position + 1

        if position == last_position:
            # Same fixing as yesterday: add one more day to the current block.
            blocks[-1][1] = blocks[-1][1] + 1
        else:
            # A new fixing: start a new block with one day.
            blocks.append([rates[position], 1])
            last_position = position

        day = day + pd.Timedelta(days=1)

    return blocks


def sr1_rate(fixings, start, end):
    """Model SR1 rate: the day-weighted average of SOFR over [start, end].

    Args:
        fixings: SOFR fixing path in percent.
        start: first day of the contract month.
        end: last day of the contract month (included).

    Returns:
        The average rate, in percent.
    """
    blocks = get_rate_blocks(fixings, start, end)

    weighted_sum = 0.0
    total_days = 0
    for rate, days in blocks:
        weighted_sum = weighted_sum + rate * days
        total_days = total_days + days

    return weighted_sum / total_days


def sr3_rate(fixings, start, end):
    """Model SR3 rate: SOFR compounded over [start, end], actual/360.

    Args:
        fixings: SOFR fixing path in percent.
        start: first day of the reference quarter.
        end: last day of the reference quarter (included), which is the day
            before the closing IMM date.

    Returns:
        The compounded rate, in percent.
    """
    blocks = get_rate_blocks(fixings, start, end)

    # A is the growth factor of 1 dollar invested for the whole period.
    accumulation = 1.0
    total_days = 0
    for rate, days in blocks:
        growth = 1 + days * rate / (100 * DAY_COUNT_BASIS)
        accumulation = accumulation * growth
        total_days = total_days + days

    # Turn the total growth into an annualized rate (simple, actual/360).
    return (accumulation - 1) * DAY_COUNT_BASIS / total_days * 100


def sr1_price(fixings, start, end):
    """Model SR1 futures price: 100 minus the SR1 rate."""
    return 100.0 - sr1_rate(fixings, start, end)


def sr3_price(fixings, start, end):
    """Model SR3 futures price: 100 minus the SR3 rate."""
    return 100.0 - sr3_rate(fixings, start, end)


def contract_price(root, fixings, start, end):
    """Model price of one contract. `root` is "SR1" or "SR3"."""
    if root == "SR1":
        return sr1_price(fixings, start, end)
    if root == "SR3":
        return sr3_price(fixings, start, end)
    raise ValueError(f"unknown product root {root!r}; expected 'SR1' or 'SR3'")


def price_contracts(
    fixings,
    contracts,
    root_col="root",
    start_col="reference_start",
    end_col="reference_end",
):
    """Price every contract in a table using one fixing path.

    This is the step the curve fit will repeat: one candidate path goes in,
    one model price per contract comes out. The default column names match the
    data panel written by `sofr_data`.

    Returns:
        A pandas Series of model prices, with the same index as `contracts`.
    """
    prices = []
    for i in range(len(contracts)):
        root = contracts[root_col].iloc[i]
        start = contracts[start_col].iloc[i]
        end = contracts[end_col].iloc[i]
        prices.append(contract_price(root, fixings, start, end))

    return pd.Series(prices, index=contracts.index, name="model_price", dtype=float)
