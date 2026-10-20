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
