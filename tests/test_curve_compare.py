"""Tests for curve_compare (Issue #6)."""

import pandas as pd
import pytest

from curve_compare.compare import compare_at_date, default_obs_dates, implied_paths_for_plot, run_comparison, sr1_strip_as_of, summarize_by_regime
from curve_compare.full_model import recover_full_model_path
from curve_compare.synthetic import generate_synthetic_world, generate_sr1_contracts


def test_full_model_recovers_true_path_from_noiseless_sr1_prices():
    """With no SOFR noise, no price noise, and no spread, the recovered deltas
    must equal the true policy moves exactly (Section 12 of docs/METHODOLOGY.md:
    "generate prices from a known path, then recover the path")."""
    world = generate_synthetic_world(sofr_noise_bps=0.0, price_noise_bps=0.0, spread_pct=0.0, seed=0)

    obs_date = world.meeting_dates[2] + pd.Timedelta(days=3)
    strip = sr1_strip_as_of(world.sr1_contracts, obs_date, months_ahead=9)
    baseline = float(world.true_policy.loc[obs_date])

    deltas = recover_full_model_path(strip, world.meeting_dates, baseline, obs_date, world.daily_sofr)

    future_meetings = [m for m in world.meeting_dates if m > obs_date and m <= strip["month_end"].max()]
    assert len(future_meetings) > 0
    for meeting in future_meetings:
        true_delta = world.move_at(meeting) / 100.0
        assert deltas[meeting] == pytest.approx(true_delta, abs=1e-9)


def test_post_meeting_anchor_keeps_next_forecast_and_plot_on_current_rate():
    world = generate_synthetic_world(sofr_noise_bps=0.0, price_noise_bps=0.0, spread_pct=0.0, seed=0)
    obs_date = world.meeting_dates[2] + pd.Timedelta(days=1)

    result = compare_at_date(world, obs_date)
    paths = implied_paths_for_plot(world, obs_date, horizon_days=50)

    assert result["full_model_error_bps"] == pytest.approx(0.0, abs=1e-9)
    assert paths.loc[obs_date, "full_model"] == pytest.approx(world.true_policy.loc[obs_date], abs=1e-9)
    assert paths.loc[result["next_meeting"], "full_model"] == pytest.approx(
        world.true_policy.loc[result["next_meeting"]], abs=1e-9
    )


def test_elapsed_sr1_fixings_are_used_for_late_month_forecasts_and_paths():
    world = generate_synthetic_world(n_meetings=2, moves_bps=[50, 0], sofr_noise_bps=0.0, spread_pct=0.0)
    obs_date = pd.Timestamp("2024-01-30")
    world.daily_sofr.loc["2024-01-26":obs_date] = 4.1
    world.sr1_contracts = generate_sr1_contracts(
        world.daily_sofr, world.sr1_contracts["month_start"], price_noise_bps=0.0
    )

    result = compare_at_date(world, obs_date)
    paths = implied_paths_for_plot(world, obs_date, horizon_days=3)

    assert result["full_model_error_bps"] == pytest.approx(0.0, abs=1e-9)
    assert paths.loc[world.meeting_dates[0], "full_model"] == pytest.approx(4.5, abs=1e-9)


def test_synthetic_history_covers_default_observations_for_early_month_meeting():
    world = generate_synthetic_world(start="2024-03-20", sofr_noise_bps=0.0)
    first_obs_date = default_obs_dates(world)[0]

    assert first_obs_date in world.daily_sofr.index
    assert pd.notna(compare_at_date(world, first_obs_date)["full_model_error_bps"])


def test_simplified_curve_error_is_larger_around_meetings_with_a_rate_change():
    """The headline check for Issue #6: a smooth curve that ignores FOMC timing
    should do noticeably worse when the next meeting actually moves the rate
    than when it is a hold, while the full model should not show nearly as
    large a gap."""
    world = generate_synthetic_world(seed=0)
    results = run_comparison(world, default_obs_dates(world))
    by_regime = summarize_by_regime(results)

    assert by_regime.loc["change", "simple_curve_mae_bps"] > by_regime.loc["hold", "simple_curve_mae_bps"]
    # The gap should be large, not just statistically nonzero: this is the
    # property the comparison exists to demonstrate.
    assert by_regime.loc["change", "simple_curve_mae_bps"] > 5 * by_regime.loc["hold", "simple_curve_mae_bps"]


def test_full_model_more_accurate_than_simplified_curve_overall():
    """Sanity check: explicitly modeling FOMC timing should reduce next-meeting
    forecast error overall, not just around rate changes."""
    world = generate_synthetic_world(seed=0)
    results = run_comparison(world, default_obs_dates(world))

    full_mae = results["full_model_error_bps"].abs().mean()
    simple_mae = results["simple_curve_error_bps"].abs().mean()
    assert full_mae < simple_mae


def test_sr1_strip_as_of_only_includes_unexpired_contracts():
    world = generate_synthetic_world(seed=0)
    obs_date = world.meeting_dates[5]
    strip = sr1_strip_as_of(world.sr1_contracts, obs_date, months_ahead=4)
    assert len(strip) == 4
    assert (strip["month_end"] > obs_date).all()
    assert strip["month_start"].is_monotonic_increasing


def test_next_meeting_after_raises_past_the_last_meeting():
    from curve_compare.compare import next_meeting_after

    world = generate_synthetic_world(seed=0)
    with pytest.raises(ValueError):
        next_meeting_after(world.meeting_dates, world.meeting_dates[-1])
