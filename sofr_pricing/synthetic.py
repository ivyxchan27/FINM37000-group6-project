"""SYNTHETIC inputs for testing and demonstrating the #4 fit. Not real data.

PLACEHOLDER: these stand in for the real tables until they are available:
- `make_synthetic_decisions` stands in for #2's `fomc_meetings.csv` (PR #13);
- `make_synthetic_market` stands in for one day of #1's settlement panel (PR #16);
- `make_synthetic_realized_sofr` stands in for published daily SOFR, which no
  issue collects yet.

The market prices are generated with the SAME pricer the fit uses. A
noiseless fit therefore recovers the moves almost exactly. That checks the
code, not the model ("inverse crime"): real prices will not fit this well.
"""

import numpy as np
import pandas as pd

from sofr_pricing.estimation import policy_level_on
from sofr_pricing.fixing_calendar import effective_date, fixing_dates
from sofr_pricing.paths import step_fixings
from sofr_pricing.pricing import price_contracts


def make_synthetic_decisions(meeting_dates, changes_bps, start_lower=5.25, start_upper=5.50):
    """A FAKE meeting table in the shape of #2's fomc_meetings.csv.

    Args:
        meeting_dates: announcement dates.
        changes_bps: the decision at each meeting, in bp (0 for a hold).
        start_lower, start_upper: the target range before the first meeting (percent).
    """
    if len(meeting_dates) != len(changes_bps):
        raise ValueError("need one change per meeting date")

    rows = []
    lower, upper = start_lower, start_upper
    for meeting, change in zip(pd.to_datetime(list(meeting_dates)), changes_bps):
        new_lower = lower + change / 100
        new_upper = upper + change / 100
        if change > 0:
            decision = "hike"
        elif change < 0:
            decision = "cut"
        else:
            decision = "hold"
        rows.append({"meeting_date": meeting,
                     "target_lower_before": lower, "target_upper_before": upper,
                     "target_lower_after": new_lower, "target_upper_after": new_upper,
                     "change_bps": float(change), "decision": decision})
        lower, upper = new_lower, new_upper
    return pd.DataFrame(rows)


def make_synthetic_realized_sofr(start, end, decisions, basis=0.0, holidays=()):
    """FAKE published SOFR: the policy level in force each day plus a constant basis."""
    dates = fixing_dates(start, end, holidays)
    rates = [policy_level_on(day, decisions, after_announcement=False) + basis for day in dates]
    return pd.Series(rates, index=dates, name="sofr")


def make_synthetic_market(obs_date, contracts, expected_moves_bps, anchor, basis=0.0,
                          known_moves=None, realized_fixings=None, noise_bps=0.0,
                          tick=None, seed=0, holidays=()):
    """FAKE settlement prices for one day, built from a chosen "market-expected" path.

    Args:
        obs_date: observation date.
        contracts: contract table (root, reference_start, reference_end), for
            example from `contract_strip`.
        expected_moves_bps: {announcement_date: expected move in bp}. These are
            the "true" answers the fit should recover.
        anchor, basis, known_moves: as in `fit_meeting_moves`.
        realized_fixings: if given, days before obs_date use these (as the
            real market would).
        noise_bps: standard deviation of random price noise, in bp.
        tick: if given, round prices to this grid (an illustration; check the
            CME specification for the real tick sizes).
        seed: random seed, so the same inputs always give the same prices.

    Returns:
        A copy of `contracts` with a settlement_price column.
    """
    from sofr_pricing.fixing_calendar import last_known_fixing_date
    from sofr_pricing.paths import splice_realized

    obs_date = pd.Timestamp(obs_date)
    table = contracts.copy()
    for column in ("reference_start", "reference_end"):
        table[column] = pd.to_datetime(table[column])

    first_day = table["reference_start"].min() - pd.Timedelta(days=10)
    last_day = table["reference_end"].max() + pd.Timedelta(days=10)
    dates = fixing_dates(first_day, last_day, holidays)

    moves = {} if known_moves is None else dict(known_moves)
    for announcement, move_bps in expected_moves_bps.items():
        if pd.Timestamp(announcement) >= dates[-1]:
            continue  # takes effect after every contract has ended: no effect on prices
        effective = effective_date(announcement, dates)
        moves[effective] = moves.get(effective, 0.0) + move_bps / 100

    path = step_fixings(dates, anchor, moves, basis)
    if realized_fixings is not None:
        path = splice_realized(realized_fixings, path, last_known_fixing_date(obs_date, dates))

    prices = price_contracts(path, table).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    prices = prices + rng.normal(0.0, noise_bps / 100, size=len(prices))
    if tick is not None:
        prices = np.round(prices / tick) * tick

    table["settlement_price"] = prices
    return table
