"""SYNTHETIC stand-ins for the outputs of #2 and #4. Not real data.

PLACEHOLDER: `make_synthetic_implied_paths` stands in for #4's implied-path
table until #4 fits real SR1/SR3 prices. Once it does, pass #4's table to
`build_forecast_errors` instead; nothing else changes. Likewise,
`make_synthetic_decisions` stands in for #2's `fomc_meetings.csv`.

The fake implied path is built FROM the realized path, plus a bias and noise
that we choose. The evaluation should then recover exactly what we injected:
with no noise, every error equals the bias; with noise, the bias is the mean
error and the RMSE is close to sqrt(bias^2 + noise^2). The tests rely on this.
"""

import numpy as np
import pandas as pd

from forecast_eval.inputs import IMPLIED_CUM, MEETING_DATE, OBS_DATE, REALIZED_CHANGE, validate_realized_decisions


def make_synthetic_decisions(start="2022-01-26", n_meetings=38, weeks_apart=6):
    """A FAKE meeting schedule and decision path, in the realized-decision shape.

    Meetings are `weeks_apart` weeks apart from `start` (the real schedule is
    irregular, roughly 8 meetings a year). The moves loosely mimic a hiking
    cycle, a long hold, then cuts. Not real FOMC decisions: the real table
    comes from #2.
    """
    dates = pd.date_range(start, periods=n_meetings, freq=f"{weeks_apart}W-WED")
    pattern = [0, 25, 50, 75, 75, 75, 75, 50, 25, 25, 25, 0, 25] + [0] * 8 + [-50, -25, -25] + [0] * 5
    pattern += [-25, -25, -25] + [0] * n_meetings  # pad so any n_meetings works
    return pd.DataFrame({MEETING_DATE: dates, REALIZED_CHANGE: np.array(pattern[:n_meetings], dtype=float)})


def make_synthetic_implied_paths(realized, obs_dates=None, max_horizon=6,
                                 bias_bps=0.0, noise_bps=0.0, seed=0):
    """PLACEHOLDER for #4's output: a FAKE implied-path table built from a known realized path.

    For every observation date and each of its next `max_horizon` meetings
    (strictly after the observation date), the implied cumulative change is

        realized cumulative change through that meeting + bias[h] + noise

    where h is how many meetings ahead it is, and noise ~ Normal(0, noise[h]),
    drawn independently for each row with a fixed seed.

    Parameters
    ----------
    realized : realized-decision table (meeting_date, change_bps).
    obs_dates : observation dates; default is every business day from 60
        calendar days before the first meeting to the day before the last one.
    max_horizon : number of meetings ahead to generate.
    bias_bps : one number for all horizons, or a list of `max_horizon` numbers
        (bias at horizon 1, 2, ...).
    noise_bps : standard deviation of the noise, same format as bias_bps.
    seed : random seed, so the same inputs always give the same table.

    Returns a DataFrame in the implied-path shape (obs_date, meeting_date,
    implied_cum_change_bps).
    """
    realized = validate_realized_decisions(realized)
    bias = _per_horizon(bias_bps, max_horizon, "bias_bps")
    noise = _per_horizon(noise_bps, max_horizon, "noise_bps")
    if (noise < 0).any():
        raise ValueError("noise_bps must be non-negative")

    schedule = realized[MEETING_DATE].to_numpy()
    cum0 = np.concatenate([[0.0], realized[REALIZED_CHANGE].cumsum().to_numpy()])
    if obs_dates is None:
        obs_dates = pd.bdate_range(schedule[0] - np.timedelta64(60, "D"), schedule[-1] - np.timedelta64(1, "D"))
    obs_dates = pd.DatetimeIndex(obs_dates).normalize()

    rng = np.random.default_rng(seed)
    rows = []
    for obs_date in obs_dates:
        first = np.searchsorted(schedule, np.datetime64(obs_date), side="right")
        for h in range(1, max_horizon + 1):
            k = first + h - 1
            if k >= len(schedule):
                break
            truth = cum0[k + 1] - cum0[first]
            rows.append({
                OBS_DATE: obs_date,
                MEETING_DATE: pd.Timestamp(schedule[k]),
                IMPLIED_CUM: truth + bias[h - 1] + rng.normal(0.0, noise[h - 1]),
            })
    return pd.DataFrame(rows, columns=[OBS_DATE, MEETING_DATE, IMPLIED_CUM])


def _per_horizon(value, max_horizon, name):
    """Turn one number or a list of numbers into an array with one value per horizon."""
    if np.ndim(value) == 0:
        return np.full(max_horizon, float(value))
    values = np.asarray(value, dtype=float)
    if values.shape != (max_horizon,):
        raise ValueError(f"{name} must be one number or a list of {max_horizon} numbers")
    return values
