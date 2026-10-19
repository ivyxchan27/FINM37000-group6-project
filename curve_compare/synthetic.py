"""Synthetic SOFR / FOMC data (Issue #6), standing in for real data pending #1/#4.

PLACEHOLDER. Everything in this module is a seeded, fully-known fake world:
a true policy-rate path that only moves on FOMC meeting dates, a daily SOFR
series built from it, and a strip of SR1 futures settlement prices built
from that SOFR series. It replaces the real Databento SR1 settlements and
the real FOMC calendar (#1/#2/#4) until those are available, at which point
`generate_synthetic_world` is replaced by real data loaders; `full_model.py`
and `simple_curve.py` only consume the `SyntheticWorld` fields below (daily
SOFR, SR1 contracts, meeting dates), so neither needs to change.

Rates are in PERCENT everywhere (4.25 means 4.25%), matching `sofr_pricing`
(PR #12). Meeting moves are in basis points (25 = a 25 bp hike), matching
`forecast_eval` (PR #15) and `fomc_data` (PR #13).

Because the true path is known exactly, the "market" SR1 prices below are
built with perfect foresight of the whole contract month (realized daily
SOFR for every day in the month, not a forecast). That is a standard way to
validate a curve-fitting method (see Section 12 of docs/METHODOLOGY.md,
PR #11: "generate prices from a known path, then recover the path") and
makes the recovery test in `tests/test_curve_compare.py` possible.

For simplicity, a meeting's new rate takes effect on the meeting's own
announcement date rather than the next business day (T_j in Section 2 of
docs/METHODOLOGY.md). This only affects the synthetic generator; it does not
change the comparison, which only needs an internally consistent true path.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from curve_compare.sr1_local import sr1_average_rate

# A realistic hike-then-hold-then-cut cycle: 5 hikes, 6 holds, 5 cuts, in bp.
DEFAULT_MOVES_BPS = [25, 25, 50, 50, 25, 0, 0, 0, 0, 0, 0, -25, -25, -50, -25, -25]


def generate_meeting_dates(start="2024-01-31", n_meetings=16, meetings_per_year=8):
    """`n_meetings` FOMC decision dates, evenly spaced to average `meetings_per_year` a year.

    The real schedule is irregular; this is close enough to space 16 meetings
    over about two years (the real FOMC meets roughly every six weeks).
    """
    start = pd.Timestamp(start)
    spacing_days = round(365 / meetings_per_year)
    return [start + pd.Timedelta(days=spacing_days * i) for i in range(n_meetings)]


def true_daily_policy_path(dates, meeting_dates, moves_bps, base_rate_pct):
    """The true policy rate on every date in `dates`: a step function that only
    moves on `meeting_dates`, by `moves_bps` (in bp), starting from `base_rate_pct`.
    """
    if len(meeting_dates) != len(moves_bps):
        raise ValueError("meeting_dates and moves_bps must be the same length")
    dates = pd.DatetimeIndex(dates)
    path = pd.Series(base_rate_pct, index=dates, dtype=float)
    cumulative_pct = 0.0
    for meeting, move_bps in zip(meeting_dates, moves_bps):
        cumulative_pct += move_bps / 100.0
        path.loc[path.index >= pd.Timestamp(meeting)] = base_rate_pct + cumulative_pct
    return path


def generate_daily_sofr(true_policy, spread_pct=0.03, noise_bps=2.0, seed=0):
    """Daily SOFR = true policy rate + a small constant spread + seeded noise.

    One value per CALENDAR day (real SOFR is only fixed on business days and
    carried over weekends; that distinction does not matter for this
    comparison, so it is skipped here for simplicity).
    """
    rng = np.random.default_rng(seed)
    noise_pct = rng.normal(0.0, noise_bps / 100.0, size=len(true_policy))
    return true_policy + spread_pct + noise_pct


def generate_sr1_contracts(daily_sofr, months, price_noise_bps=0.0, seed=1):
    """SR1 settlement price for each calendar month in `months`.

    price = 100 - (day-weighted average daily SOFR over the month), optionally
    with a small amount of independent seeded noise added to the settlement
    price itself (representing quotation noise, separate from the daily SOFR
    noise above).

    Returns a DataFrame with columns [month_start, month_end, price].
    """
    rng = np.random.default_rng(seed)
    rows = []
    for month_start in months:
        month_start = pd.Timestamp(month_start).replace(day=1)
        month_end = month_start + pd.offsets.MonthEnd(0)
        rate = sr1_average_rate(daily_sofr, month_start, month_end)
        price = 100.0 - rate
        if price_noise_bps:
            price += rng.normal(0.0, price_noise_bps / 100.0)
        rows.append({"month_start": month_start, "month_end": month_end, "price": price})
    return pd.DataFrame(rows)


@dataclass
class SyntheticWorld:
    """One fully-known synthetic market: true path, daily SOFR, and an SR1 strip."""

    meeting_dates: list
    moves_bps: pd.Series  # indexed by meeting date
    base_rate_pct: float
    spread_pct: float
    true_policy: pd.Series  # daily, in percent
    daily_sofr: pd.Series  # daily, in percent
    sr1_contracts: pd.DataFrame  # columns: month_start, month_end, price

    def move_at(self, meeting_date):
        """The realized move (bp) at `meeting_date`, or 0.0 if it is not a meeting date."""
        return float(self.moves_bps.get(pd.Timestamp(meeting_date), 0.0))


def generate_synthetic_world(
    start="2024-01-31",
    n_meetings=16,
    meetings_per_year=8,
    moves_bps=None,
    base_rate_pct=4.50,
    spread_pct=0.03,
    sofr_noise_bps=2.0,
    price_noise_bps=0.0,
    seed=0,
    extra_months_after_last_meeting=9,
):
    """Build one seeded synthetic world (see module docstring).

    `extra_months_after_last_meeting` controls how far past the last meeting
    the daily SOFR series and SR1 strip extend, so that observation dates
    near the end of the schedule still have a full strip of forward-looking
    SR1 contracts to fit (see `compare.sr1_strip_as_of`).
    """
    moves_bps = DEFAULT_MOVES_BPS if moves_bps is None else moves_bps
    if len(moves_bps) != n_meetings:
        raise ValueError(f"moves_bps must have length n_meetings ({n_meetings})")

    meeting_dates = generate_meeting_dates(start, n_meetings, meetings_per_year)
    moves = pd.Series(moves_bps, index=pd.DatetimeIndex(meeting_dates), dtype=float)

    span_start = pd.Timestamp(meeting_dates[0]).replace(day=1)
    last_month = pd.Timestamp(meeting_dates[-1]) + pd.DateOffset(months=extra_months_after_last_meeting)
    span_end = last_month + pd.offsets.MonthEnd(0)
    dates = pd.date_range(span_start, span_end, freq="D")

    true_policy = true_daily_policy_path(dates, meeting_dates, moves_bps, base_rate_pct)
    daily_sofr = generate_daily_sofr(true_policy, spread_pct, sofr_noise_bps, seed)

    months = pd.date_range(span_start, span_end, freq="MS")
    sr1_contracts = generate_sr1_contracts(daily_sofr, months, price_noise_bps, seed=seed + 1)

    return SyntheticWorld(
        meeting_dates=meeting_dates,
        moves_bps=moves,
        base_rate_pct=base_rate_pct,
        spread_pct=spread_pct,
        true_policy=true_policy,
        daily_sofr=daily_sofr,
        sr1_contracts=sr1_contracts,
    )
