"""Full (FOMC-aware) model: recover a meeting-by-meeting step path from SR1 prices.

Scoped-down version of the fit in Section 5 of docs/METHODOLOGY.md (PR #11)
for SR1 only: the unknowns are the rate changes `delta_j` at each FOMC
meeting date, and they are recovered one contract month at a time (a
"bootstrap"), using the day-weighted average equation from Section 3.

Within one SR1 contract month [month_start, month_end], future days use

    rate(d) = baseline + sum(delta_j for every meeting j with month_start <= j <= d)

where `baseline` is the observed anchor rate plus every delta resolved in
earlier months. In the observation month, elapsed days instead use their
observed SOFR fixings. Only meetings strictly after the observation date are
unknown; setting the sum of observed and future rates equal to the SR1
day-weighted average gives ONE linear equation in the month's unknown deltas:

    delta * days_from_meeting_to_month_end = avg_rate * days_in_month
        - sum(observed fixings through the observation date)
        - baseline * remaining_days

With exactly one meeting in the month this has one unknown and solves
exactly, which is the usual case (meetings are about six weeks apart).
"""

import pandas as pd


def anchor_sofr_rate(daily_sofr, obs_date, meeting_dates, window_days=5):
    """A short trailing average of `daily_sofr` since the latest meeting.

    Used as the bootstrap's starting level `baseline` instead of the single
    day's SOFR print, so that one day's idiosyncratic noise does not get
    carried, uncorrected, into every later recovered delta (the bootstrap is
    sequential, so an error in the anchor propagates to every later month).
    docs/METHODOLOGY.md Section 2 similarly fixes the anchor to "the observed
    policy level," i.e. data, not a free parameter; a short trailing average
    is a simple stand-in for that here.
    """
    obs_date = pd.Timestamp(obs_date)
    latest_meeting = max((pd.Timestamp(m) for m in meeting_dates if pd.Timestamp(m) <= obs_date), default=None)
    observations = daily_sofr.loc[:obs_date] if latest_meeting is None else daily_sofr.loc[latest_meeting:obs_date]
    window = observations.tail(window_days)
    return float(window.mean())


def recover_full_model_path(sr1_contracts, meeting_dates, baseline_rate, obs_date, daily_sofr):
    """Bootstrap the rate change at every meeting covered by `sr1_contracts`.

    Parameters
    ----------
    sr1_contracts : DataFrame with columns [month_start, month_end, price],
        sorted or not, covering the months to fit (see `compare.sr1_strip_as_of`).
    meeting_dates : every scheduled FOMC meeting date (past and future).
    baseline_rate : the anchor rate in force at `obs_date` (see `anchor_sofr_rate`).
    obs_date : the observation date; only meetings strictly after it are unknown.
    daily_sofr : observed daily SOFR fixings through `obs_date`.

    Returns
    -------
    dict mapping each resolved meeting date (a `pd.Timestamp` strictly after
    `obs_date`, falling within the contracts' date range) to its recovered
    delta, in the same units as `baseline_rate` (percent).

    Raises
    ------
    ValueError : if a contract month contains multiple unresolved meetings;
        one monthly average cannot identify their separate moves.
    """
    obs_date = pd.Timestamp(obs_date)
    meeting_dates = sorted(pd.Timestamp(m) for m in meeting_dates)
    contracts = sr1_contracts.sort_values("month_start")

    deltas = {}
    baseline = baseline_rate
    for _, row in contracts.iterrows():
        month_start, month_end, price = row["month_start"], row["month_end"], row["price"]
        avg_target = 100.0 - price
        days_in_month = (month_end - month_start).days + 1

        breakpoints = [m for m in meeting_dates if month_start <= m <= month_end and m > obs_date]
        if not breakpoints:
            continue  # nothing new to resolve in this contract month
        if len(breakpoints) != 1:
            raise ValueError("SR1 contract month must contain exactly one unresolved meeting")

        meeting = breakpoints[0]
        days_after_meeting = (month_end - meeting).days + 1
        observed = daily_sofr.loc[month_start:obs_date] if month_start <= obs_date else daily_sofr.iloc[:0]
        observed_days = (obs_date - month_start).days + 1 if month_start <= obs_date else 0
        if len(observed) != observed_days:
            raise ValueError("missing observed daily SOFR fixings in SR1 contract month")
        remaining_days = days_in_month - observed_days
        delta = (avg_target * days_in_month - observed.sum() - baseline * remaining_days) / days_after_meeting
        deltas[meeting] = delta
        baseline += delta

    return deltas


def implied_daily_path(dates, baseline_rate, deltas):
    """Daily implied rate: `baseline_rate` plus every resolved delta whose meeting has occurred.

    `deltas` is the output of `recover_full_model_path`. Dates before the
    first resolved meeting are flat at `baseline_rate`.
    """
    dates = pd.DatetimeIndex(dates)
    rate = pd.Series(baseline_rate, index=dates, dtype=float)
    for meeting, delta in sorted(deltas.items()):
        rate.loc[rate.index >= meeting] += delta
    return rate
