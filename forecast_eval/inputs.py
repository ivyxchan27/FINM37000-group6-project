"""Input tables for the forecast-accuracy evaluation (Issue #5), and checks on them.

The evaluation joins two tables. Both use basis points (bp) for rate changes
(25 means a 25 bp hike, -50 a 50 bp cut).

1. Implied paths (from #4)
--------------------------
One row per (observation date, FOMC meeting):

    obs_date                 trading day the futures prices were observed
    meeting_date             FOMC announcement date of an upcoming meeting
    implied_cum_change_bps   implied CUMULATIVE change in the policy rate, in bp,
                             from the policy rate in force on obs_date through
                             this meeting (that is, the sum of the implied moves
                             at every meeting after obs_date up to and including
                             this one)

This is the "table of implied meeting moves and cumulative changes by horizon"
planned in #4. If #4 also outputs the per-meeting moves, only the cumulative
column is needed here; the per-meeting move is recovered as the difference
between consecutive cumulative values.

Rows whose meeting_date is on or before obs_date are dropped (see
`validate_implied_paths`): on a decision day the futures settle after the
announcement, so that day's price is a nowcast of the decision, not a forecast.

2. Realized decisions (from #2)
-------------------------------
One row per scheduled FOMC meeting:

    meeting_date   FOMC announcement date
    change_bps     realized change in the target-range midpoint at that meeting, in bp

These are the column names already used by `fomc_meetings.csv` from #2 (the
output of `fomc_data.dataset.build_fomc_decisions`), so that table can be
passed in unchanged; its extra columns (decision, target bounds, ...) are
ignored. Any other source needs at most a rename onto these two names.

The table must list EVERY scheduled meeting from the earliest observation date
onward, holds included, because "the h-th meeting ahead" is counted on this
schedule.
"""

import pandas as pd

# Column names of the implied-path table.
OBS_DATE = "obs_date"
MEETING_DATE = "meeting_date"
IMPLIED_CUM = "implied_cum_change_bps"

# Column names of the realized-decision table (same as #2's output).
REALIZED_CHANGE = "change_bps"

IMPLIED_COLUMNS = [OBS_DATE, MEETING_DATE, IMPLIED_CUM]
REALIZED_COLUMNS = [MEETING_DATE, REALIZED_CHANGE]


def _check_columns(df, required, table_name):
    """Raise a clear error if `df` is not a DataFrame with the required columns."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{table_name} must be a pandas DataFrame, got {type(df).__name__}")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing columns {missing}; expected {required}")
    if len(df) == 0:
        raise ValueError(f"{table_name} is empty")


def _to_dates(series, table_name):
    """Convert a column to dates (time of day dropped), with a clear error on bad values."""
    try:
        dates = pd.to_datetime(series)
    except (ValueError, TypeError) as err:
        raise ValueError(f"{table_name}: column '{series.name}' has values that are not dates") from err
    if dates.isna().any():
        raise ValueError(f"{table_name}: column '{series.name}' has missing dates")
    return dates.dt.normalize()


def _to_numbers(series, table_name):
    """Convert a column to floats, with a clear error on bad or missing values."""
    numbers = pd.to_numeric(series, errors="coerce")
    if numbers.isna().any():
        raise ValueError(f"{table_name}: column '{series.name}' has missing or non-numeric values")
    return numbers.astype(float)


def validate_realized_decisions(realized):
    """Check the realized-decision table and return a clean copy.

    Returns a DataFrame with columns [meeting_date, change_bps], one row per
    meeting, sorted by meeting date. Extra columns are dropped.
    """
    _check_columns(realized, REALIZED_COLUMNS, "realized decisions")
    out = pd.DataFrame({
        MEETING_DATE: _to_dates(realized[MEETING_DATE], "realized decisions"),
        REALIZED_CHANGE: _to_numbers(realized[REALIZED_CHANGE], "realized decisions"),
    })
    if out[MEETING_DATE].duplicated().any():
        dupes = out.loc[out[MEETING_DATE].duplicated(), MEETING_DATE].dt.date.tolist()
        raise ValueError(f"realized decisions: meeting dates appear more than once: {dupes}")
    return out.sort_values(MEETING_DATE).reset_index(drop=True)


def validate_implied_paths(implied, realized):
    """Check the implied-path table against the meeting schedule and return a clean copy.

    `realized` must already be validated (see `validate_realized_decisions`);
    its meeting dates are the schedule.

    Returns a DataFrame with columns [obs_date, meeting_date, implied_cum_change_bps],
    sorted by (obs_date, meeting_date), after dropping:
      - rows whose meeting is on or before obs_date (a decision-day price is a
        nowcast, not a forecast), and
      - rows for meetings outside the schedule's date range: after its last
        meeting (the outcome is not known yet) or before its first meeting
        (no realized decision to score against).

    Raises ValueError if a meeting date falls inside the schedule's date range
    but is not one of its meetings. That almost always means the two tables
    use different date conventions (for example the effective date, the day
    after the announcement, instead of the announcement date itself).
    """
    _check_columns(implied, IMPLIED_COLUMNS, "implied paths")
    out = pd.DataFrame({
        OBS_DATE: _to_dates(implied[OBS_DATE], "implied paths"),
        MEETING_DATE: _to_dates(implied[MEETING_DATE], "implied paths"),
        IMPLIED_CUM: _to_numbers(implied[IMPLIED_CUM], "implied paths"),
    })

    if out.duplicated([OBS_DATE, MEETING_DATE]).any():
        raise ValueError("implied paths: some (obs_date, meeting_date) pairs appear more than once")

    # Decision-day exclusion: keep only meetings strictly after the observation date.
    out = out[out[MEETING_DATE] > out[OBS_DATE]]

    # Meetings outside the schedule's date range cannot be scored.
    schedule = realized[MEETING_DATE]
    out = out[(out[MEETING_DATE] >= schedule.min()) & (out[MEETING_DATE] <= schedule.max())]

    unknown = out.loc[~out[MEETING_DATE].isin(schedule), MEETING_DATE].unique()
    if len(unknown) > 0:
        shown = [pd.Timestamp(d).date() for d in unknown[:5]]
        raise ValueError(
            f"implied paths: meeting dates {shown} are not in the realized-decision schedule; "
            "both tables must use the FOMC announcement date"
        )

    return out.sort_values([OBS_DATE, MEETING_DATE]).reset_index(drop=True)
