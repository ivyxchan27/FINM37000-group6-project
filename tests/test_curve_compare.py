"""Tests for curve_compare (Issue #6)."""

import pandas as pd
import pytest

from curve_compare.compare import default_obs_dates, run_comparison, sr1_strip_as_of, summarize_by_regime
from curve_compare.full_model import recover_full_model_path
from curve_compare.synthetic import generate_synthetic_world


def test_full_model_recovers_true_path_from_noiseless_sr1_prices():
    """With no SOFR noise, no price noise, and no spread, the recovered deltas
    must equal the true policy moves exactly (Section 12 of docs/METHODOLOGY.md:
    "generate prices from a known path, then recover the path")."""
    world = generate_synthetic_world(sofr_noise_bps=0.0, price_noise_bps=0.0, spread_pct=0.0, seed=0)

    obs_date = world.meeting_dates[2] + pd.Timedelta(days=3)
    strip = sr1_strip_as_of(world.sr1_contracts, obs_date, months_ahead=9)
    baseline = float(world.true_policy.loc[obs_date])

    deltas = recover_full_model_path(strip, world.meeting_dates, baseline, obs_date)

    future_meetings = [m for m in world.meeting_dates if m > obs_date and m <= strip["month_end"].max()]
    assert len(future_meetings) > 0
    for meeting in future_meetings:
        true_delta = world.move_at(meeting) / 100.0
        assert deltas[meeting] == pytest.approx(true_delta, abs=1e-9)


def test_full_model_recovery_handles_two_meetings_in_one_contract_month():
    """When a contract month holds two still-unresolved meetings, the single
    day-weighted equation is underdetermined; `np.linalg.lstsq` should still
    return deltas that reproduce the observed SR1 price exactly, even though
    it cannot separate the two meetings' individual sizes."""
    meeting_dates = [pd.Timestamp("2024-03-05"), pd.Timestamp("2024-03-20")]
    sr1_contracts = pd.DataFrame([
        {"month_start": pd.Timestamp("2024-03-01"), "month_end": pd.Timestamp("2024-03-31"), "price": 95.0},
    ])
    obs_date = pd.Timestamp("2024-02-15")
    baseline = 4.5

    deltas = recover_full_model_path(sr1_contracts, meeting_dates, baseline, obs_date)

    days_in_month = 31
    days_from = {m: (pd.Timestamp("2024-03-31") - m).days + 1 for m in meeting_dates}
    reconstructed_avg = baseline + sum(deltas[m] * days_from[m] for m in meeting_dates) / days_in_month
    assert reconstructed_avg == pytest.approx(5.0, abs=1e-9)  # 100 - price = 5.0


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
