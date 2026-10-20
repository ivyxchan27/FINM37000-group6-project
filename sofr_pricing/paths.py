"""Build candidate SOFR fixing paths from FOMC rate moves (Issue #4).

Units: rates and moves are in PERCENT, so a 25 bp cut is -0.25.

The model path follows docs/METHODOLOGY.md (Section 2):

    SOFR(day) = anchor + basis + (sum of every move whose effective date <= day)

- `anchor` is the policy rate in force today. It is DATA, not something we fit.
- `basis` is the gap between SOFR and the policy rate. We hold it fixed.
- The moves are the unknowns that the curve fit in `estimation.py` searches for.
"""

import pandas as pd


def step_fixings(dates, anchor, moves, basis=0.0):
    """Build a SOFR fixing path that steps at each FOMC effective date.

    Args:
        dates: fixing dates, for example from `fixing_dates`.
        anchor: policy rate before any of the moves, in percent.
        moves: dictionary {effective_date: change_in_percent}.
            Example: {"2024-09-19": -0.50} is a 50 bp cut effective Sep 19.
        basis: SOFR minus policy rate, in percent, held constant.

    Returns:
        A pandas Series of SOFR rates indexed by `dates`.

    Example:
        >>> from sofr_pricing.fixing_calendar import fixing_dates
        >>> d = fixing_dates("2024-06-17", "2024-06-21")
        >>> step_fixings(d, 5.00, {"2024-06-20": -0.25}).tolist()
        [5.0, 5.0, 5.0, 4.75, 4.75]
    """
    path = pd.Series(float(anchor + basis), index=dates, name="sofr")

    for effective, change in moves.items():
        # The move applies from its effective date onwards.
        path[path.index >= pd.Timestamp(effective)] += change

    return path


def splice_realized(realized, model, last_realized):
    """Combine published SOFR (the past) with a model path (the future).

    Days up to and including `last_realized` come from `realized`; later days
    come from `model`. This is how the already-known part of an in-progress
    contract month enters the price.

    Args:
        realized: published SOFR fixings in percent (a Series indexed by date).
        model: candidate model path in percent.
        last_realized: the last fixing date whose SOFR is known
            (see `last_known_fixing_date`).

    Raises:
        ValueError: if `realized` has no fixings on or before `last_realized`.
    """
    cutoff = pd.Timestamp(last_realized)

    past = realized[realized.index <= cutoff]
    if len(past) == 0:
        raise ValueError(f"no realized fixings on or before {cutoff.date()}")

    future = model[model.index > cutoff]
    return pd.concat([past, future]).sort_index()
