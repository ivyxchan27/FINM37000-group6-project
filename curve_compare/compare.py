"""Compare the full model and the simplified curve at the next-meeting horizon (Issue #6).

"Next-meeting horizon" matches the horizon-1 definition in `forecast_eval`
(PR #15, Issue #5): for an observation date `t`, the forecast target is the
first scheduled FOMC meeting STRICTLY after `t` (a meeting's own decision
day is never its own horizon-1 target, since that day's price is a nowcast,
not a forecast).

Errors are measured against the TRUE POLICY rate, not the daily SOFR rate.
Both models are fit in SOFR-rate space (because SR1 prices are built from
SOFR, not the policy rate directly), so the known constant spread
(`world.spread_pct`) is subtracted from each model's implied rate before
comparing to the truth. In a real analysis the spread would have to be
estimated (see Section 2 of docs/METHODOLOGY.md, PR #11); here it is a
synthetic-data parameter, so it is simply known.
"""

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from curve_compare.full_model import anchor_sofr_rate, implied_daily_path, recover_full_model_path
from curve_compare.simple_curve import simple_curve_path

METHODS = ("full_model", "simple_curve")

# Skips US federal holidays (not just weekends) so a window starting near
# January 1st does not land on New Year's Day itself.
_FEDERAL_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def sr1_strip_as_of(sr1_contracts, obs_date, months_ahead=9):
    """The SR1 contracts still "listed" (not yet expired) as of `obs_date`, nearest first."""
    obs_date = pd.Timestamp(obs_date)
    strip = sr1_contracts[sr1_contracts["month_end"] > obs_date]
    return strip.sort_values("month_start").head(months_ahead).reset_index(drop=True)


def next_meeting_after(meeting_dates, obs_date):
    """The first scheduled meeting strictly after `obs_date` (the horizon-1 target)."""
    obs_date = pd.Timestamp(obs_date)
    future = sorted(m for m in meeting_dates if pd.Timestamp(m) > obs_date)
    if not future:
        raise ValueError(f"no scheduled meeting after {obs_date.date()}")
    return future[0]


def default_obs_dates(world):
    """Business days from just before the first meeting to just before the last.

    Stops before the last meeting so that `next_meeting_after` can always
    find a real meeting (with a known realized outcome) to score against.
    Uses a US-federal-holiday business-day calendar, not just weekdays, so a
    window starting near January 1st begins on the first real business day
    of the year rather than on New Year's Day itself.
    """
    start = world.meeting_dates[0] - pd.Timedelta(days=30)
    end = world.meeting_dates[-1] - pd.Timedelta(days=1)
    return pd.date_range(start, end, freq=_FEDERAL_BUSINESS_DAY)


def compare_at_date(world, obs_date, months_ahead=9, anchor_window_days=5):
    """Run both methods at one observation date; return one result dict."""
    obs_date = pd.Timestamp(obs_date)
    strip = sr1_strip_as_of(world.sr1_contracts, obs_date, months_ahead)
    next_meeting = next_meeting_after(world.meeting_dates, obs_date)

    baseline = anchor_sofr_rate(world.daily_sofr, obs_date, world.meeting_dates, anchor_window_days)
    deltas = recover_full_model_path(strip, world.meeting_dates, baseline, obs_date, world.daily_sofr)
    full_rate_sofr = float(implied_daily_path([next_meeting], baseline, deltas).iloc[0])

    simple_rate_sofr = float(simple_curve_path(strip, obs_date, [next_meeting]).iloc[0])

    true_rate_policy = float(world.true_policy.loc[next_meeting])
    full_rate_policy = full_rate_sofr - world.spread_pct
    simple_rate_policy = simple_rate_sofr - world.spread_pct

    return {
        "obs_date": obs_date,
        "next_meeting": next_meeting,
        "days_to_meeting": (next_meeting - obs_date).days,
        "rate_change_at_meeting": world.move_at(next_meeting) != 0.0,
        "full_model_error_bps": (full_rate_policy - true_rate_policy) * 100.0,
        "simple_curve_error_bps": (simple_rate_policy - true_rate_policy) * 100.0,
    }


def run_comparison(world, obs_dates=None, months_ahead=9, anchor_window_days=5):
    """`compare_at_date` for every date in `obs_dates` (default: `default_obs_dates(world)`)."""
    obs_dates = default_obs_dates(world) if obs_dates is None else obs_dates
    rows = [compare_at_date(world, d, months_ahead, anchor_window_days) for d in obs_dates]
    return pd.DataFrame(rows)


def _method_stats(err):
    return {
        "n_obs": len(err),
        "bias_bps": err.mean(),
        "mae_bps": err.abs().mean(),
        "rmse_bps": np.sqrt((err**2).mean()),
    }


def summarize_errors(results):
    """One row per method: n_obs, bias, MAE, RMSE (bp), at the next-meeting horizon."""
    rows = {method: _method_stats(results[f"{method}_error_bps"]) for method in METHODS}
    return pd.DataFrame(rows).T


def method_difference_bps(results):
    """simple_curve error minus full_model error, in bp, for each observation."""
    return results["simple_curve_error_bps"] - results["full_model_error_bps"]


def summarize_by_regime(results):
    """Accuracy by whether the forecasted meeting was a rate change or a hold.

    This is the headline check for Issue #6: does the simplified curve do
    worse specifically around meetings where the rate actually moved?
    """
    regime = np.where(results["rate_change_at_meeting"], "change", "hold")
    rows = []
    for name, group in results.assign(regime=regime).groupby("regime"):
        row = {"regime": name, "n_obs": len(group)}
        for method in METHODS:
            row.update({f"{method}_{k}": v for k, v in _method_stats(group[f"{method}_error_bps"]).items() if k != "n_obs"})
        rows.append(row)
    return pd.DataFrame(rows).set_index("regime")


def weekly_bins(results, width_days=7):
    """Evenly spaced bucket edges (0, 7, 14, ...) covering every days_to_meeting."""
    max_days = int(results["days_to_meeting"].max())
    n_buckets = int(np.ceil(max_days / width_days))
    return [width_days * i for i in range(n_buckets + 1)]


def summarize_by_days_to_meeting(results, width_days=7):
    """Accuracy bucketed by calendar days before the meeting, for both methods.

    Used to check whether the simplified curve's errors concentrate near
    FOMC meetings (see the #7 chart in `plots.plot_error_by_days_to_meeting`).
    """
    edges = weekly_bins(results, width_days)
    labels = [f"{lo + 1}-{hi}" for lo, hi in zip(edges[:-1], edges[1:])]
    bucket = pd.cut(results["days_to_meeting"], bins=edges, labels=labels)
    rows = []
    for label, group in results.groupby(bucket, observed=True):
        if len(group) == 0:
            continue
        row = {"days_bucket": label, "n_obs": len(group)}
        for method in METHODS:
            row.update({f"{method}_{k}": v for k, v in _method_stats(group[f"{method}_error_bps"]).items() if k != "n_obs"})
        rows.append(row)
    return pd.DataFrame(rows).set_index("days_bucket")


def implied_paths_for_plot(world, obs_date, horizon_days=240, months_ahead=9, anchor_window_days=5):
    """Daily true/full-model/simple-curve rate paths from `obs_date` forward, for plotting."""
    obs_date = pd.Timestamp(obs_date)
    dates = pd.date_range(obs_date, periods=horizon_days, freq="D")
    strip = sr1_strip_as_of(world.sr1_contracts, obs_date, months_ahead)

    baseline = anchor_sofr_rate(world.daily_sofr, obs_date, world.meeting_dates, anchor_window_days)
    deltas = recover_full_model_path(strip, world.meeting_dates, baseline, obs_date, world.daily_sofr)
    full_path = implied_daily_path(dates, baseline, deltas) - world.spread_pct
    simple_path = simple_curve_path(strip, obs_date, dates) - world.spread_pct
    true_path = world.true_policy.reindex(dates)

    return pd.DataFrame({"true": true_path, "full_model": full_path, "simple_curve": simple_path})
