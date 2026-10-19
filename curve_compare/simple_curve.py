"""Simplified SOFR curve, ignoring FOMC timing, ported from the course package.

Source: `build_short_rate_curve` in
`src/finm37000/futures.py` of https://github.com/pattersonem/finm37000-autumn-2026
(the course package referenced in the team README's "Comparison with a
Simplified Approach" section). That function bootstraps a continuously
compounded zero-rate curve from one date's strip of STIR futures settlement
prices: it treats each contract's quoted rate as a constant simple forward
rate over the incremental period since the previous contract's expiration,
compounds those incremental discount factors in sequence, and returns a
function that linearly interpolates (flat-extrapolated) the resulting zero
rates between contract maturities.

It is reimplemented here, not imported, for two reasons: this package must
not depend on `databento` at runtime (course package dependency, not
allowed here), and the course function is written for a general STIR
strip (SR1 or SR3) read from a live `databento` client, whereas this
version only needs one date's SR1 strip, already in memory as a DataFrame.
The bootstrap arithmetic (lines below from `discount_factor /= ...` to
`-np.log(discount_factor) / t`) is copied as closely as the different input
shape allows; `days_per_year=360.0` matches that function's own default.

What makes this curve "simplified" is exactly what it leaves out: nothing
here looks at FOMC meeting dates. Each node's maturity is just its
contract's month-end, and the zero rate is linearly interpolated between
nodes. The zero curve is continuous, but its derived forward rate can jump
at a contract boundary; it has no meeting-date breakpoint unless a meeting
coincides with that boundary. This lack of FOMC timing is the property this
comparison (Issue #6) is testing.
"""

import numpy as np
import pandas as pd


def build_simple_zero_curve(sr1_contracts, obs_date, days_per_year=360.0):
    """Bootstrap a smooth zero-rate curve from one date's SR1 strip.

    Parameters
    ----------
    sr1_contracts : DataFrame with columns [month_start, month_end, price],
        the SR1 strip observed as of `obs_date` (see `compare.sr1_strip_as_of`).
    obs_date : the date the curve is built as of (the maturity anchor, t=0).
    days_per_year : day-count basis for years-to-expiration, matching the
        course function's own default.

    Returns
    -------
    (curve_t, curve_r) : two 1-D numpy arrays, sorted by maturity. `curve_t`
        is years to each contract's month-end; `curve_r` is the bootstrapped
        continuously-compounded zero rate, in PERCENT, at that maturity.
    """
    obs_date = pd.Timestamp(obs_date)
    contracts = sr1_contracts.sort_values("month_end").copy()
    contracts["years_to_expiration"] = (contracts["month_end"] - obs_date).dt.days / days_per_year
    contracts = contracts[contracts["years_to_expiration"] > 0]
    if len(contracts) < 2:
        raise ValueError("need at least 2 SR1 contracts after obs_date to build a curve")

    # The course convention quotes rates as a decimal fraction: 100 - price.
    forward_rate = (100.0 - contracts["price"]) / 100.0

    curve_t, curve_r = [], []
    discount_factor = 1.0
    prior_t = 0.0
    for t, rate in zip(contracts["years_to_expiration"], forward_rate):
        discount_factor /= 1.0 + rate * (t - prior_t)
        curve_t.append(t)
        curve_r.append(-np.log(discount_factor) / t * 100.0)  # back to percent
        prior_t = t

    return np.array(curve_t), np.array(curve_r)


def simple_curve_daily_path(curve_t, curve_r, obs_date, dates, days_per_year=360.0):
    """Implied daily SOFR rate from the zero curve, ignoring FOMC timing.

    `curve_t`/`curve_r` are the output of `build_simple_zero_curve`. The
    returned value at each date is the local instantaneous FORWARD rate
    implied by the zero curve: the derivative of the cumulative curve
    `t * zero_rate(t)`. Because `curve_r` is linearly interpolated in
    maturity, this derivative is just `zero_rate(t) + t * slope(t)`, where
    `slope(t)` is the interpolant's slope in the segment containing `t` (zero
    outside the curve's first/last node, matching `np.interp`'s flat
    extrapolation). This is what makes the result a smooth curve rather than
    a meeting-date step function: a forward rate derived this way can jump
    at a contract boundary, but has no FOMC-specific breakpoint.

    Returns a daily `pd.Series` of rates in percent, indexed by `dates`.
    """
    if len(curve_t) < 2:
        raise ValueError("need at least 2 curve nodes to interpolate a forward rate")
    obs_date = pd.Timestamp(obs_date)
    dates = pd.DatetimeIndex(dates)
    t = (dates - obs_date).days.to_numpy() / days_per_year

    zero_rate = np.interp(t, curve_t, curve_r)

    idx = np.clip(np.searchsorted(curve_t, t) - 1, 0, len(curve_t) - 2)
    slope = (curve_r[idx + 1] - curve_r[idx]) / (curve_t[idx + 1] - curve_t[idx])
    inside = (t > curve_t[0]) & (t < curve_t[-1])
    forward_rate = zero_rate + t * np.where(inside, slope, 0.0)

    return pd.Series(forward_rate, index=dates, name="simple_curve_rate")


def simple_curve_path(sr1_contracts, obs_date, dates, days_per_year=360.0):
    """Convenience wrapper: build the zero curve and evaluate it on `dates` in one call."""
    curve_t, curve_r = build_simple_zero_curve(sr1_contracts, obs_date, days_per_year)
    return simple_curve_daily_path(curve_t, curve_r, obs_date, dates, days_per_year)
