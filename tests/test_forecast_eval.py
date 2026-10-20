"""Unit tests for the forecast-accuracy evaluation (Issue #5).

How to run (from the repository root):
    python -m pytest tests/test_forecast_eval.py -v

All data here is synthetic. Expected numbers are worked out by hand in the
comments, or are the bias and noise we injected into the fake implied paths.
"""

import matplotlib

matplotlib.use("Agg")  # no window needed for the plotting test

import numpy as np
import pandas as pd
import pytest

from forecast_eval import (
    build_forecast_errors,
    classify_direction,
    error_by_days_to_meeting,
    hit_rate_sensitivity,
    make_synthetic_decisions,
    make_synthetic_implied_paths,
    plot_error_by_days_to_meeting,
    plot_error_by_horizon,
    summarize_by_horizon,
    summarize_by_regime,
    validate_implied_paths,
    validate_realized_decisions,
    weekly_day_bins,
)

# ---------------------------------------------------------------------------
# A tiny hand-checkable example
# ---------------------------------------------------------------------------
# Three meetings: +25 on Mar 16, hold on May 4, +50 on Jun 15.
REALIZED = pd.DataFrame({
    "meeting_date": pd.to_datetime(["2022-03-16", "2022-05-04", "2022-06-15"]),
    "change_bps": [25.0, 0.0, 50.0],
})


def implied_rows(rows):
    """Build an implied-path table from (obs_date, meeting_date, cum_bps) tuples."""
    return pd.DataFrame(rows, columns=["obs_date", "meeting_date", "implied_cum_change_bps"]).assign(
        obs_date=lambda d: pd.to_datetime(d["obs_date"]),
        meeting_date=lambda d: pd.to_datetime(d["meeting_date"]),
    )


def test_hand_example_errors():
    # On Mar 1 the market expects +20 at Mar 16, then +5 more by May 4, then
    # +30 more by Jun 15 (cumulative 20, 25, 55).
    # Realized cumulative: 25, 25, 75. Errors: -5, 0, -20.
    implied = implied_rows([
        ("2022-03-01", "2022-03-16", 20.0),
        ("2022-03-01", "2022-05-04", 25.0),
        ("2022-03-01", "2022-06-15", 55.0),
    ])
    errors = build_forecast_errors(implied, REALIZED)

    assert errors["horizon"].tolist() == [1, 2, 3]
    assert errors["realized_cum_bps"].tolist() == [25.0, 25.0, 75.0]
    assert errors["error_bps"].tolist() == [-5.0, 0.0, -20.0]
    assert errors["days_to_meeting"].tolist() == [15, 64, 106]
    # Own moves: implied 20, 5, 30 vs realized 25, 0, 50.
    assert errors["implied_move_bps"].tolist() == [20.0, 5.0, 30.0]
    assert errors["realized_move_bps"].tolist() == [25.0, 0.0, 50.0]
    assert errors["implied_direction"].tolist() == ["hike", "hold", "hike"]
    assert errors["direction_hit"].tolist() == [True, True, True]


def test_decision_day_is_not_a_forecast():
    # On Mar 16 (decision day) the Mar 16 meeting is a nowcast and is dropped;
    # horizon 1 is May 4. Realized cumulative from Mar 16: May 4 -> 0, Jun 15 -> 50.
    implied = implied_rows([
        ("2022-03-16", "2022-03-16", 25.0),  # nowcast: must be ignored
        ("2022-03-16", "2022-05-04", 3.0),
        ("2022-03-16", "2022-06-15", 40.0),
    ])
    errors = build_forecast_errors(implied, REALIZED)

    assert errors["meeting_date"].tolist() == list(pd.to_datetime(["2022-05-04", "2022-06-15"]))
    assert errors["horizon"].tolist() == [1, 2]
    assert errors["realized_cum_bps"].tolist() == [0.0, 50.0]
    assert errors["error_bps"].tolist() == [3.0, -10.0]


def test_summary_statistics_by_hand():
    # Errors -5, 0, -20 at horizons 1, 2, 3 (from test_hand_example_errors), plus a
    # second day (Mar 2) with errors +5, 0, -10 at the same horizons.
    implied = implied_rows([
        ("2022-03-01", "2022-03-16", 20.0), ("2022-03-01", "2022-05-04", 25.0), ("2022-03-01", "2022-06-15", 55.0),
        ("2022-03-02", "2022-03-16", 30.0), ("2022-03-02", "2022-05-04", 25.0), ("2022-03-02", "2022-06-15", 65.0),
    ])
    table = summarize_by_horizon(build_forecast_errors(implied, REALIZED))

    # Horizon 1: errors -5, +5 -> bias 0, MAE 5, RMSE 5.
    assert table.loc[1, "bias_bps"] == pytest.approx(0.0)
    assert table.loc[1, "mae_bps"] == pytest.approx(5.0)
    assert table.loc[1, "rmse_bps"] == pytest.approx(5.0)
    # Horizon 3: errors -20, -10 -> bias -15, MAE 15, RMSE sqrt((400 + 100) / 2).
    assert table.loc[3, "bias_bps"] == pytest.approx(-15.0)
    assert table.loc[3, "mae_bps"] == pytest.approx(15.0)
    assert table.loc[3, "rmse_bps"] == pytest.approx(np.sqrt(250.0))
    assert table.loc[3, "n_obs"] == 2
    assert table.loc[3, "n_meetings"] == 1
    # Mar 2, horizon 2: implied own move 25 - 30 = -5 -> hold, realized hold: hit.
    assert table.loc[2, "hit_rate"] == pytest.approx(1.0)


def test_direction_threshold():
    assert classify_direction(12.6) == "hike"
    assert classify_direction(12.5) == "hold"  # exactly on the threshold counts as hold
    assert classify_direction(-12.6) == "cut"
    assert classify_direction(-30, threshold_bps=50) == "hold"
    assert list(classify_direction([20.0, np.nan, -20.0])) == ["hike", None, "cut"]


def test_direction_miss_counts_against_hit_rate():
    # Market expects only +10 at Mar 16 (a "hold"), but the Fed hiked 25: a miss.
    implied = implied_rows([("2022-03-01", "2022-03-16", 10.0)])
    errors = build_forecast_errors(implied, REALIZED)
    assert errors.loc[0, "implied_direction"] == "hold"
    assert errors.loc[0, "direction_hit"] == False  # noqa: E712 (value is a numpy/pandas bool)
    assert summarize_by_horizon(errors).loc[1, "hit_rate"] == 0.0


def test_hit_rate_sensitivity_by_threshold():
    # Implied own moves 20, 5, 30 vs realized 25, 0, 50 (as in test_hand_example_errors),
    # plus a second day with own moves 8 and 4 at horizons 1 and 2.
    implied = implied_rows([
        ("2022-03-01", "2022-03-16", 20.0), ("2022-03-01", "2022-05-04", 25.0), ("2022-03-01", "2022-06-15", 55.0),
        ("2022-03-02", "2022-03-16", 8.0), ("2022-03-02", "2022-05-04", 12.0),
    ])
    table = hit_rate_sensitivity(build_forecast_errors(implied, REALIZED))
    assert table.columns.tolist() == ["hit_rate_5bp", "hit_rate_12.5bp", "hit_rate_20bp"]
    # Horizon 1 (realized hike): own moves 20 and 8.
    #   5 bp: both hikes -> 1.0;  12.5 bp: 20 hike, 8 hold -> 0.5;  20 bp: both holds -> 0.0.
    assert table.loc[1].tolist() == [1.0, 0.5, 0.0]
    # Horizon 2 (realized hold): own moves 5 and 4 -> hold at every threshold.
    assert table.loc[2].tolist() == [1.0, 1.0, 1.0]
    # Horizon 3 (realized +50): own move 30 -> hike at every threshold.
    assert table.loc[3].tolist() == [1.0, 1.0, 1.0]


def test_hit_rate_sensitivity_matches_main_table_at_default():
    realized = make_synthetic_decisions()
    errors = build_forecast_errors(make_synthetic_implied_paths(realized, noise_bps=10.0, seed=3), realized)
    np.testing.assert_allclose(hit_rate_sensitivity(errors)["hit_rate_12.5bp"],
                               summarize_by_horizon(errors)["hit_rate"])


def test_missing_previous_meeting_leaves_direction_unknown():
    # Horizon 2 is present but horizon 1 is not, so the own move at May 4 is unknown.
    implied = implied_rows([("2022-03-01", "2022-05-04", 25.0)])
    errors = build_forecast_errors(implied, REALIZED)
    assert errors["horizon"].tolist() == [2]
    assert errors.loc[0, "error_bps"] == 0.0
    assert pd.isna(errors.loc[0, "direction_hit"])
    assert np.isnan(summarize_by_horizon(errors).loc[2, "hit_rate"])


# ---------------------------------------------------------------------------
# Recovering what we injected into the synthetic stand-in for #4
# ---------------------------------------------------------------------------

def test_noiseless_synthetic_recovers_bias_exactly():
    realized = make_synthetic_decisions()
    bias = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    implied = make_synthetic_implied_paths(realized, bias_bps=bias, noise_bps=0.0)
    table = summarize_by_horizon(build_forecast_errors(implied, realized))

    # With no noise every error equals the bias, so bias = MAE = RMSE = injected value.
    assert table.index.tolist() == [1, 2, 3, 4, 5, 6]
    np.testing.assert_allclose(table["bias_bps"], bias)
    np.testing.assert_allclose(table["mae_bps"], bias)
    np.testing.assert_allclose(table["rmse_bps"], bias)


def test_perfect_forecast_has_zero_error_and_full_hit_rate():
    realized = make_synthetic_decisions()
    implied = make_synthetic_implied_paths(realized)
    table = summarize_by_horizon(build_forecast_errors(implied, realized))
    np.testing.assert_allclose(table["mae_bps"], 0.0, atol=1e-12)
    np.testing.assert_allclose(table["hit_rate"], 1.0)


def test_noisy_synthetic_recovers_bias_and_noise():
    realized = make_synthetic_decisions()
    implied = make_synthetic_implied_paths(realized, bias_bps=5.0, noise_bps=10.0, seed=1)
    table = summarize_by_horizon(build_forecast_errors(implied, realized))

    # About 1,000 draws per horizon: the mean error is within ~1 bp of the
    # injected 5 bp (standard error 10 / sqrt(1000) ~ 0.3 bp), and RMSE is
    # close to sqrt(5^2 + 10^2) ~ 11.2 bp.
    np.testing.assert_allclose(table["bias_bps"], 5.0, atol=1.5)
    np.testing.assert_allclose(table["rmse_bps"], np.sqrt(125.0), atol=1.0)


def test_synthetic_is_reproducible():
    realized = make_synthetic_decisions()
    a = make_synthetic_implied_paths(realized, noise_bps=5.0, seed=7)
    b = make_synthetic_implied_paths(realized, noise_bps=5.0, seed=7)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Days-to-meeting and regime tables, and the chart
# ---------------------------------------------------------------------------

def test_days_to_meeting_buckets():
    implied = implied_rows([
        ("2022-03-10", "2022-03-16", 20.0),  # 6 days ahead, error -5
        ("2022-03-01", "2022-03-16", 15.0),  # 15 days ahead, error -10
    ])
    table = error_by_days_to_meeting(build_forecast_errors(implied, REALIZED), bins=[0, 7, 14, 1000])
    # The empty 8-14 bucket is left out.
    assert table.index.tolist() == ["1-7", "15+"]
    assert table.loc["1-7", "mae_bps"] == 5.0
    assert table.loc["15+", "mae_bps"] == 10.0


def test_weekly_bins_are_evenly_spaced_and_cover_all_days():
    implied = implied_rows([
        ("2022-03-10", "2022-03-16", 20.0),  # 6 days ahead
        ("2022-03-01", "2022-03-16", 15.0),  # 15 days ahead
    ])
    errors = build_forecast_errors(implied, REALIZED)
    # 15 days needs three 7-day buckets: (0,7], (7,14], (14,21].
    assert weekly_day_bins(errors) == [0, 7, 14, 21]
    table = error_by_days_to_meeting(errors, bins=weekly_day_bins(errors), open_ended=False)
    assert table.index.tolist() == ["1-7", "15-21"]


def test_weekly_chart_has_one_point_per_nonempty_week():
    realized = make_synthetic_decisions()
    errors = build_forecast_errors(make_synthetic_implied_paths(realized, noise_bps=5.0), realized)
    fig = plot_error_by_days_to_meeting(errors)
    ax = fig.axes[0]
    expected = error_by_days_to_meeting(errors, bins=weekly_day_bins(errors), open_ended=False)
    assert [t.get_text() for t in ax.get_xticklabels()] == expected.index.tolist()
    assert len(ax.lines[0].get_xdata()) == len(expected)


def test_regime_split():
    realized = make_synthetic_decisions()
    implied = make_synthetic_implied_paths(realized, bias_bps=2.0)
    errors = build_forecast_errors(implied, realized)
    regimes = {"early": ("2022-01-01", "2023-06-30"), "late": ("2023-07-01", "2026-12-31")}
    table = summarize_by_regime(errors, regimes=regimes)

    assert table.index.get_level_values("regime").unique().tolist() == ["early", "late"]
    np.testing.assert_allclose(table["bias_bps"], 2.0)
    # Every synthetic meeting is inside one of the two regimes, so no row is dropped.
    assert table["n_obs"].sum() == len(errors)


def test_regime_follows_meeting_date_not_obs_date():
    # Observed in 2022 (regime "a") but forecasting the 2023 meeting (regime "b").
    realized = pd.DataFrame({"meeting_date": pd.to_datetime(["2022-12-14", "2023-02-01"]),
                             "change_bps": [50.0, 25.0]})
    implied = implied_rows([
        ("2022-12-01", "2022-12-14", 50.0),  # horizon 1: 2022 meeting -> regime "a"
        ("2022-12-01", "2023-02-01", 85.0),  # horizon 2: 2023 meeting -> regime "b", error +10
    ])
    regimes = {"a": ("2022-01-01", "2022-12-31"), "b": ("2023-01-01", "2023-12-31")}
    table = summarize_by_regime(build_forecast_errors(implied, realized), regimes=regimes)
    assert table.index.tolist() == [("a", 1), ("b", 2)]
    assert table.loc[("b", 2), "bias_bps"] == 10.0


def test_overlapping_regimes_rejected():
    errors = build_forecast_errors(implied_rows([("2022-03-01", "2022-03-16", 20.0)]), REALIZED)
    with pytest.raises(ValueError, match="overlaps"):
        summarize_by_regime(errors, regimes={"a": ("2022-01-01", "2022-12-31"), "b": ("2022-06-01", "2023-12-31")})


def test_plot_returns_figure():
    realized = make_synthetic_decisions()
    errors = build_forecast_errors(make_synthetic_implied_paths(realized, noise_bps=5.0), realized)
    fig = plot_error_by_horizon(errors)
    assert len(fig.axes) == 2


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_fomc_data_columns_need_no_rename():
    # #2's fomc_meetings.csv has these columns (and more); it is accepted as is.
    fomc_meetings = REALIZED.assign(decision=["hike", "hold", "hike"], effective_date=pd.NaT)
    out = validate_realized_decisions(fomc_meetings)
    assert out.columns.tolist() == ["meeting_date", "change_bps"]


def test_missing_column_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        validate_realized_decisions(REALIZED.rename(columns={"change_bps": "move"}))


def test_duplicate_meeting_rejected():
    with pytest.raises(ValueError, match="more than once"):
        validate_realized_decisions(pd.concat([REALIZED, REALIZED.iloc[[0]]]))


def test_non_numeric_change_rejected():
    bad = REALIZED.assign(change_bps=["25", "oops", "50"])
    with pytest.raises(ValueError, match="non-numeric"):
        validate_realized_decisions(bad)


def test_duplicate_implied_rows_rejected():
    implied = implied_rows([("2022-03-01", "2022-03-16", 20.0), ("2022-03-01", "2022-03-16", 21.0)])
    with pytest.raises(ValueError, match="more than once"):
        validate_implied_paths(implied, validate_realized_decisions(REALIZED))


def test_wrong_date_convention_rejected():
    # Mar 17 is the effective date, not the announcement date: a convention mismatch.
    implied = implied_rows([("2022-03-01", "2022-03-17", 25.0)])
    with pytest.raises(ValueError, match="not in the realized-decision schedule"):
        build_forecast_errors(implied, REALIZED)


def test_meetings_outside_schedule_are_dropped():
    # A meeting after the last realized one has no known outcome yet; it is skipped.
    implied = implied_rows([("2022-06-01", "2022-06-15", 50.0), ("2022-06-01", "2022-07-27", 120.0)])
    errors = build_forecast_errors(implied, REALIZED)
    assert errors["meeting_date"].tolist() == [pd.Timestamp("2022-06-15")]


def test_no_overlap_raises_clear_error():
    implied = implied_rows([("2022-07-01", "2022-07-27", 75.0)])  # after the schedule ends
    with pytest.raises(ValueError, match="no forecast errors"):
        summarize_by_horizon(build_forecast_errors(implied, REALIZED))


def test_open_ended_last_bucket_catches_all():
    # bins=[0, 7, 14, 21, 28] with open_ended=True: the last edge is a
    # catch-all, so a forecast 42 days out lands in "22+", not dropped.
    errors = build_forecast_errors(implied_rows([("2022-05-04", "2022-06-15", 60.0)]), REALIZED)
    assert errors["days_to_meeting"].tolist() == [42]
    table = error_by_days_to_meeting(errors, bins=[0, 7, 14, 21, 28])
    # Buckets with no forecasts are left out of the table.
    assert table.index.tolist() == ["22+"]
    assert table["n_obs"].tolist() == [1]


def test_days_outside_closed_bins_raise():
    # open_ended=False: every bucket must have both edges, so a forecast past
    # the last edge is a caller error, not a silently missing row.
    errors = build_forecast_errors(implied_rows([("2022-05-04", "2022-06-15", 60.0)]), REALIZED)
    with pytest.raises(ValueError, match="outside the supplied bin edges"):
        error_by_days_to_meeting(errors, bins=[0, 7, 14, 21, 28], open_ended=False)
