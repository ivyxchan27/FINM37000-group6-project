"""Forecast errors and accuracy statistics by horizon (Issue #5).

Definitions (they follow Section 9 of docs/METHODOLOGY.md, PR #11):

Horizon
    For observation date t, horizon h is the h-th scheduled FOMC meeting
    STRICTLY AFTER t. A meeting is never its own horizon-1 forecast on its
    decision day: futures settle after the announcement, so that day's price
    is a nowcast. On that day horizon 1 is the following meeting.

Forecast and realized value
    implied cumulative change through the h-th meeting (C_hat in the
    methodology) versus the realized cumulative change through the same
    meeting (C_actual): the sum of the realized changes at every meeting after
    t up to and including the h-th one. This equals the methodology's
    L_after_m_h - L_t when L_t is the policy level in force after any meeting
    on or before t.

Forecast error, in bp
    error = implied - realized. Positive error: the market expected a higher
    policy rate than the Fed delivered (it over-predicted hikes or
    under-predicted cuts).

Statistics per horizon
    bias = mean error, MAE = mean absolute error, RMSE = root mean squared error.

Direction hit
    Uses the h-th meeting's OWN move, not the cumulative change: the implied
    move at that meeting (implied cumulative through h minus implied cumulative
    through h-1) and the realized change at that meeting are each labelled
    hike (> +threshold), cut (< -threshold) or hold (otherwise). A hit is
    when the two labels agree. The default threshold is 12.5 bp, half of a
    standard 25 bp move: an implied move of -12.5 bp is a 50% chance of a
    25 bp cut, so it is the point where "cut" becomes more likely than "hold".
"""

import numpy as np
import pandas as pd

from forecast_eval.inputs import (
    IMPLIED_CUM,
    MEETING_DATE,
    OBS_DATE,
    REALIZED_CHANGE,
    validate_implied_paths,
    validate_realized_decisions,
)

MAX_HORIZON = 6
DIRECTION_THRESHOLD_BPS = 12.5
# Thresholds for the hit-rate sensitivity check. Futures may carry a risk
# premium that shifts implied moves away from pure expectations, so we check
# that the hit rate does not hinge on the 12.5 bp choice.
SENSITIVITY_THRESHOLDS_BPS = (5.0, 12.5, 20.0)

# Regimes for the split table, by the date of the meeting being forecast.
# Inclusive (start, end) dates.
DEFAULT_REGIMES = {
    "2022-23 hikes": ("2022-01-01", "2023-12-31"),
    "2024-25 cuts": ("2024-01-01", "2025-12-31"),
    "2026": ("2026-01-01", "2026-12-31"),
}

# Calendar-day buckets for "error vs days before the meeting". Each bucket is
# (lower, upper], so the first one holds forecasts made 1 to 7 days ahead.
DEFAULT_DAY_BINS = [0, 7, 14, 21, 28, 42, 10_000]


def classify_direction(move_bps, threshold_bps=DIRECTION_THRESHOLD_BPS):
    """Label each move as 'hike', 'cut' or 'hold'. Missing moves stay missing.

    `move_bps` is a number or an array-like of numbers in bp.
    """
    if threshold_bps < 0:
        raise ValueError("threshold_bps must be non-negative")
    moves = np.asarray(move_bps, dtype=float)
    labels = np.where(moves > threshold_bps, "hike", np.where(moves < -threshold_bps, "cut", "hold"))
    labels = np.where(np.isnan(moves), None, labels).astype(object)
    return labels if labels.ndim else labels.item()


def build_forecast_errors(implied, realized, max_horizon=MAX_HORIZON,
                          direction_threshold_bps=DIRECTION_THRESHOLD_BPS):
    """Match implied paths with realized decisions: one row per (obs_date, horizon).

    Parameters
    ----------
    implied : DataFrame in the implied-path shape (see forecast_eval.inputs).
    realized : DataFrame in the realized-decision shape (see forecast_eval.inputs).
    max_horizon : evaluate horizons 1..max_horizon meetings ahead.
    direction_threshold_bps : threshold for the hike / hold / cut labels.

    Returns
    -------
    DataFrame with columns
        obs_date, horizon, meeting_date, days_to_meeting,
        implied_cum_bps, realized_cum_bps, error_bps,
        implied_move_bps, realized_move_bps,
        implied_direction, realized_direction, direction_hit

    A (obs_date, horizon) pair only appears when the implied table has a value
    for the h-th meeting after obs_date and that meeting is in the schedule.
    implied_move_bps (and so the direction columns) is missing when the
    implied value for meeting h-1 is missing.
    """
    if not (isinstance(max_horizon, (int, np.integer)) and max_horizon >= 1):
        raise ValueError("max_horizon must be a positive integer")

    realized = validate_realized_decisions(realized)
    implied = validate_implied_paths(implied, realized)

    schedule = realized[MEETING_DATE].to_numpy()
    # Cumulative realized change with a leading 0: cum0[i] is the total change
    # over the first i meetings, so the change over meetings i..j-1 is cum0[j] - cum0[i].
    cum0 = np.concatenate([[0.0], realized[REALIZED_CHANGE].cumsum().to_numpy()])

    # Look up the implied cumulative change by (obs_date, meeting_date).
    implied_lookup = implied.set_index([OBS_DATE, MEETING_DATE])[IMPLIED_CUM]

    rows = []
    for obs_date in implied[OBS_DATE].unique():
        # Index of the first scheduled meeting strictly after obs_date
        # (side="right" skips a meeting held on obs_date itself).
        first = np.searchsorted(schedule, np.datetime64(obs_date), side="right")
        previous_implied = 0.0  # implied cumulative change through meeting h-1 (none for h=1)

        for h in range(1, max_horizon + 1):
            k = first + h - 1  # schedule index of the h-th meeting ahead
            if k >= len(schedule):
                break  # ran past the last meeting with a known outcome
            meeting = pd.Timestamp(schedule[k])
            implied_cum = implied_lookup.get((obs_date, meeting), np.nan)
            realized_cum = cum0[k + 1] - cum0[first]

            if not np.isnan(implied_cum):
                rows.append({
                    OBS_DATE: obs_date,
                    "horizon": h,
                    MEETING_DATE: meeting,
                    "days_to_meeting": (meeting - obs_date).days,
                    "implied_cum_bps": implied_cum,
                    "realized_cum_bps": realized_cum,
                    "error_bps": implied_cum - realized_cum,
                    # The meeting's own move; missing if meeting h-1 has no implied value.
                    "implied_move_bps": implied_cum - previous_implied,
                    "realized_move_bps": cum0[k + 1] - cum0[k],
                })
            previous_implied = implied_cum

    columns = [OBS_DATE, "horizon", MEETING_DATE, "days_to_meeting", "implied_cum_bps",
               "realized_cum_bps", "error_bps", "implied_move_bps", "realized_move_bps"]
    errors = pd.DataFrame(rows, columns=columns)
    errors["implied_direction"] = classify_direction(errors["implied_move_bps"], direction_threshold_bps)
    errors["realized_direction"] = classify_direction(errors["realized_move_bps"], direction_threshold_bps)
    # A hit needs both labels; it is missing (not False) when the implied move is unknown.
    has_both = errors["implied_direction"].notna()
    errors["direction_hit"] = (errors["implied_direction"] == errors["realized_direction"]).where(has_both)
    return errors


def _accuracy_stats(group):
    """Accuracy statistics for one group of forecast errors."""
    err = group["error_bps"]
    hits = group["direction_hit"].dropna()
    return pd.Series({
        "n_obs": len(err),
        # Forecasts aimed at the same meeting share one outcome, so their errors
        # are strongly correlated: n_meetings is closer to the number of
        # independent observations than n_obs is.
        "n_meetings": group[MEETING_DATE].nunique(),
        "bias_bps": err.mean(),
        "mae_bps": err.abs().mean(),
        "rmse_bps": np.sqrt((err ** 2).mean()),
        "hit_rate": hits.astype(float).mean() if len(hits) else np.nan,
    })


def _summarize(errors, by):
    """Apply _accuracy_stats to each group of `errors` defined by the columns `by`."""
    if len(errors) == 0:
        raise ValueError("no forecast errors to summarize; check that the implied and realized tables overlap")
    table = errors.groupby(by, observed=True)[["error_bps", "direction_hit", MEETING_DATE]].apply(_accuracy_stats)
    table[["n_obs", "n_meetings"]] = table[["n_obs", "n_meetings"]].astype(int)
    return table


def summarize_by_horizon(errors):
    """Accuracy table, one row per horizon: n_obs, n_meetings, bias, MAE, RMSE (bp), hit rate."""
    return _summarize(errors, "horizon")


def hit_rate_sensitivity(errors, thresholds_bps=SENSITIVITY_THRESHOLDS_BPS):
    """Direction hit rate by horizon at several hike / hold / cut thresholds.

    Returns a table with one row per horizon and one column per threshold
    (named like "hit_rate_5bp", "hit_rate_12.5bp"). The labels are recomputed
    from the implied and realized own moves for each threshold, so the
    `direction_hit` column of `errors` (built at the default 12.5 bp) is not used.
    Rows whose implied own move is missing are left out, as in the main table.
    """
    if len(errors) == 0:
        raise ValueError("no forecast errors to summarize; check that the implied and realized tables overlap")
    known = errors.dropna(subset=["implied_move_bps"])
    table = pd.DataFrame(index=pd.Index(sorted(errors["horizon"].unique()), name="horizon"))
    for threshold in thresholds_bps:
        implied = classify_direction(known["implied_move_bps"], threshold)
        realized = classify_direction(known["realized_move_bps"], threshold)
        hits = pd.Series(implied == realized, index=known.index, dtype=float)
        table[f"hit_rate_{threshold:g}bp"] = hits.groupby(known["horizon"]).mean()
    return table


def error_by_days_to_meeting(errors, horizon=1, bins=DEFAULT_DAY_BINS, open_ended=True):
    """Accuracy at one horizon, bucketed by calendar days before the meeting.

    Shows how quickly the forecast improves as the meeting gets closer.
    `bins` are bucket edges in days; each bucket is (lower, upper]. Buckets
    with no forecasts are left out of the table. With open_ended=True the last
    bucket is labelled "N+" (its upper edge is just a catch-all); with False
    it is labelled with both edges, as for the evenly spaced weekly bins.
    """
    subset = errors[errors["horizon"] == horizon]
    if len(subset) == 0:
        raise ValueError(f"no forecast errors at horizon {horizon}")
    labels = [f"{lo + 1}-{hi}" for lo, hi in zip(bins[:-1], bins[1:])]
    if open_ended:
        labels[-1] = f"{bins[-2] + 1}+"
    buckets = pd.cut(subset["days_to_meeting"], bins=bins, labels=labels)
    return _summarize(subset.assign(days_bucket=buckets), "days_bucket")


def weekly_day_bins(errors, horizon=1, width_days=7):
    """Evenly spaced bucket edges (0, 7, 14, ...) covering every days_to_meeting at one horizon.

    Used for the #7 chart, where equal-width buckets are easier to read than
    the uneven table buckets.
    """
    days = errors.loc[errors["horizon"] == horizon, "days_to_meeting"]
    if len(days) == 0:
        raise ValueError(f"no forecast errors at horizon {horizon}")
    n_buckets = int(np.ceil(days.max() / width_days))
    return [width_days * i for i in range(n_buckets + 1)]


def summarize_by_regime(errors, regimes=None):
    """Accuracy by (regime, horizon). A row's regime is set by the date of the
    meeting being forecast (meeting_date), not the observation date.

    The outcome decides the regime: at 6 meetings ahead, a forecast made in
    late 2023 for a 2024 cut belongs with the cuts, not the hikes.
    `regimes` maps a name to inclusive (start, end) dates; the default is
    DEFAULT_REGIMES. Meetings outside every regime are left out.
    """
    regimes = DEFAULT_REGIMES if regimes is None else regimes
    # Check the date ranges: each must be in order and no two may overlap.
    spans = sorted((pd.Timestamp(start), pd.Timestamp(end), name) for name, (start, end) in regimes.items())
    for start, end, name in spans:
        if start > end:
            raise ValueError(f"regime '{name}' ends before it starts")
    for (_, prev_end, prev_name), (start, _, name) in zip(spans[:-1], spans[1:]):
        if start <= prev_end:
            raise ValueError(f"regime '{name}' overlaps regime '{prev_name}'")

    regime = pd.Series(pd.NA, index=errors.index, dtype=object)
    for name, (start, end) in regimes.items():
        regime[errors[MEETING_DATE].between(pd.Timestamp(start), pd.Timestamp(end))] = name
    # Keep the regimes in the order given rather than alphabetical order.
    regime = pd.Categorical(regime, categories=list(regimes), ordered=True)
    labelled = errors.assign(regime=regime).dropna(subset=["regime"])
    return _summarize(labelled, ["regime", "horizon"])
