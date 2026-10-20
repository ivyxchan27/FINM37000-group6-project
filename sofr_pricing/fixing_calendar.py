"""Business-day calendar for SOFR fixings.

SOFR is only published on business days. A fixing applies to every calendar
day until the next fixing, so a Friday fixing also covers Saturday and Sunday.

The default calendar is Monday to Friday with no holidays. To match the real
SOFR calendar, pass the holiday dates yourself (the SIFMA U.S. government
securities calendar). We do not hard-code a holiday list here.
"""

import pandas as pd

# The longest time one fixing can stay in force. A Friday fixing before a
# Monday holiday covers Fri, Sat, Sun and Mon, which is 4 calendar days.
# If two fixings are further apart than this, data is probably missing, and
# the pricing functions raise an error instead of silently reusing a stale rate.
MAX_CARRY_DAYS = 4


def fixing_dates(start, end, holidays=()):
    """Return the business days on which SOFR is fixed, from start to end.

    Both `start` and `end` are included (when they are business days).

    Example:
        >>> [d.strftime("%a %Y-%m-%d") for d in fixing_dates("2024-06-13", "2024-06-18")]
        ['Thu 2024-06-13', 'Fri 2024-06-14', 'Mon 2024-06-17', 'Tue 2024-06-18']
    """
    # freq="C" means "custom business day": Monday to Friday minus the holidays.
    holiday_list = [pd.Timestamp(h) for h in holidays]
    return pd.bdate_range(start, end, freq="C", holidays=holiday_list)


def effective_date(announcement, dates):
    """Return the first fixing date strictly AFTER an FOMC announcement date.

    The FOMC announces its decision in the afternoon, and the new rate applies
    from the next business day. So the first SOFR fixing that reflects the
    decision is the next fixing date.

    Example:
        >>> d = fixing_dates("2024-12-16", "2024-12-31", holidays=["2024-12-25"])
        >>> effective_date("2024-12-18", d).strftime("%a %Y-%m-%d")
        'Thu 2024-12-19'
    """
    announcement = pd.Timestamp(announcement)
    for day in dates:
        if day > announcement:
            return day
    raise ValueError(f"no fixing date after {announcement.date()}; extend the calendar")


def last_known_fixing_date(trade_date, dates):
    """Return the newest fixing date whose SOFR is already published on `trade_date`.

    Assumption: SOFR for a date is published on the NEXT business day. So when
    futures settle at the end of `trade_date`, the newest known fixing is the
    previous business day's. Confirm this against the NY Fed's publication
    schedule before relying on it for the final analysis.

    Example:
        >>> d = fixing_dates("2024-06-10", "2024-06-21")
        >>> last_known_fixing_date("2024-06-17", d).strftime("%a %Y-%m-%d")
        'Fri 2024-06-14'
    """
    trade_date = pd.Timestamp(trade_date)
    known = [day for day in dates if day < trade_date]
    if len(known) == 0:
        raise ValueError(f"no fixing date before {trade_date.date()}; extend the calendar")
    return known[-1]
