"""Tests for the market-implied Fed path fit (Issue #4).

All data here is SYNTHETIC. The market prices are made with the same pricer
the fit uses, so a noiseless fit should recover the chosen moves almost
exactly. That checks the code, not the model.

Run from the repository root:  python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from sofr_pricing import (
    contract_strip,
    estimate_basis,
    fit_many,
    fit_meeting_moves,
    policy_level_on,
    upcoming_meetings,
    fixing_dates,
)
from sofr_pricing.synthetic import (
    make_synthetic_decisions,
    make_synthetic_market,
    make_synthetic_realized_sofr,
)

# ---------------------------------------------------------------------------
# A shared synthetic setup
# ---------------------------------------------------------------------------

OBS = pd.Timestamp("2024-06-03")
ANCHOR = 5.375   # target-range midpoint, percent
BASIS = -0.06    # SOFR minus policy rate, percent

# Real 2024-25 FOMC announcement dates; the expected moves below are invented.
MEETINGS = pd.to_datetime([
    "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
])
EXPECTED_BPS = {"2024-06-12": -2.0, "2024-07-31": -6.0, "2024-09-18": -18.0,
                "2024-11-07": -12.0, "2024-12-18": -10.0, "2025-01-29": -8.0,
                "2025-03-19": -7.0}


def true_moves(result):
    """The expected move (bp) at each meeting the fit estimated."""
    return np.array([EXPECTED_BPS.get(str(m.date()), 0.0) for m in result.meetings["meeting_date"]])


def market(**kwargs):
    return make_synthetic_market(OBS, contract_strip(OBS), EXPECTED_BPS, ANCHOR, BASIS, **kwargs)


# ---------------------------------------------------------------------------
# Recovering known moves
# ---------------------------------------------------------------------------


def test_noiseless_fit_recovers_the_moves():
    result = fit_meeting_moves(OBS, market(), MEETINGS, ANCHOR, BASIS)

    errors = result.meetings["implied_move_bps"].to_numpy() - true_moves(result)
    assert np.abs(errors).max() < 1e-3                       # far below 0.001 bp
    assert result.diagnostics["converged"]
    assert result.diagnostics["rmse_bps"] < 1e-3
    assert result.diagnostics["jacobian_rank"] == result.diagnostics["n_meetings"]
    assert result.diagnostics["fully_identified"]


def test_sensitivities_match_hand_calculations():
    sens = fit_meeting_moves(OBS, market(), MEETINGS, ANCHOR, BASIS).sensitivities
    # Jun 12 meeting takes effect Jun 13: 18 of June's 30 days -> -18/30 = -0.6.
    assert sens.loc["SR1-2024-06", "2024-06-12"] == pytest.approx(-0.6, abs=1e-9)
    # The Jul 31 meeting takes effect Aug 1: no July day, every August day.
    assert sens.loc["SR1-2024-07", "2024-07-31"] == pytest.approx(0.0, abs=1e-9)
    assert sens.loc["SR1-2024-08", "2024-07-31"] == pytest.approx(-1.0, abs=1e-9)


def test_cumulative_change_is_the_running_sum_of_moves():
    result = fit_meeting_moves(OBS, market(), MEETINGS, ANCHOR, BASIS)
    moves = result.meetings["implied_move_bps"].to_numpy()
    assert result.meetings["implied_cum_change_bps"].to_numpy() == pytest.approx(np.cumsum(moves))
    # By hand: -2 - 6 - 18 = -26 bp through the third meeting.
    assert result.meetings["implied_cum_change_bps"].iloc[2] == pytest.approx(-26.0, abs=1e-3)


def test_noisy_rounded_prices_still_give_close_answers():
    # 0.25 bp price noise, then rounding to a 0.0025 grid (an illustrative tick).
    for seed in range(5):
        result = fit_meeting_moves(OBS, market(noise_bps=0.25, tick=0.0025, seed=seed),
                                   MEETINGS, ANCHOR, BASIS)
        errors = result.meetings["implied_move_bps"].to_numpy() - true_moves(result)
        well_seen = result.meetings["sensitivity"].to_numpy() > 5

        assert np.abs(errors[well_seen]).max() < 1.0          # meetings many contracts see
        assert abs(np.cumsum(errors)[5]) < 1.0                # cumulative through 6 meetings
        assert result.diagnostics["rmse_bps"] < 0.5


def test_stronger_ridge_pulls_moves_toward_zero():
    light = fit_meeting_moves(OBS, market(), MEETINGS, ANCHOR, BASIS)
    heavy = fit_meeting_moves(OBS, market(), MEETINGS, ANCHOR, BASIS, ridge=1.0)

    assert np.abs(heavy.meetings["implied_move_bps"]).sum() < np.abs(light.meetings["implied_move_bps"]).sum()
    assert heavy.diagnostics["rmse_bps"] > light.diagnostics["rmse_bps"]


def test_too_few_contracts_is_flagged_by_the_rank():
    # One July contract cannot tell apart three meetings that all move July.
    july = contract_strip(OBS, n_sr1=13, n_sr3=0).iloc[[1]]   # SR1 2024-07
    schedule = pd.to_datetime(["2024-06-12", "2024-07-10", "2024-07-24"])
    prices = make_synthetic_market(OBS, july, {"2024-07-10": -10.0}, ANCHOR, BASIS)

    result = fit_meeting_moves(OBS, prices, schedule, ANCHOR, BASIS)

    assert result.diagnostics["n_meetings"] == 3
    assert result.diagnostics["jacobian_rank"] == 1
    assert not result.diagnostics["fully_identified"]
    assert result.diagnostics["rmse_bps"] < 1e-3   # it still matches the one price

    # ...but matching the price is NOT recovering the moves: the truth is -10 bp
    # at Jul 10 only, and the fit spreads it over all three meetings.
    truth = np.array([0.0, -10.0, 0.0])
    errors = result.meetings["implied_move_bps"].to_numpy() - truth
    assert np.abs(errors).max() > 1.0


# ---------------------------------------------------------------------------
# Which meetings are unknowns
# ---------------------------------------------------------------------------


def test_a_meeting_on_the_observation_date_is_not_an_unknown():
    dates = fixing_dates("2024-06-01", "2025-12-31")
    table = upcoming_meetings("2024-06-12", MEETINGS, dates, "2025-09-16")
    assert table["meeting_date"].iloc[0] == pd.Timestamp("2024-07-31")


def test_meetings_after_the_last_contract_are_left_out():
    result = fit_meeting_moves(OBS, market(), MEETINGS, ANCHOR, BASIS)
    last_contract_day = result.contracts["reference_end"].max()
    assert (result.meetings["effective_date"] <= last_contract_day).all()


def test_a_schedule_that_stops_too_early_raises():
    short_schedule = MEETINGS[:5]  # ends Dec 2024, contracts run to Sep 2025
    with pytest.raises(ValueError, match="meeting schedule ends"):
        fit_meeting_moves(OBS, market(), short_schedule, ANCHOR, BASIS)


# ---------------------------------------------------------------------------
# Anchor, basis and realized SOFR
# ---------------------------------------------------------------------------

DECISIONS = make_synthetic_decisions(["2024-06-12", "2024-07-31", "2024-09-18"], [0, 0, -50])


def test_policy_level_before_on_and_after_a_decision_day():
    # Range 5.25-5.50 (midpoint 5.375) until the Sep 18 cut to 4.75-5.00 (4.875).
    assert policy_level_on("2024-06-03", DECISIONS) == pytest.approx(5.375)
    assert policy_level_on("2024-09-18", DECISIONS, after_announcement=True) == pytest.approx(4.875)
    assert policy_level_on("2024-09-18", DECISIONS, after_announcement=False) == pytest.approx(5.375)
    assert policy_level_on("2024-09-19", DECISIONS, after_announcement=False) == pytest.approx(4.875)


def test_estimate_basis_recovers_a_constant_spread_even_across_a_cut():
    realized = make_synthetic_realized_sofr("2024-09-01", "2024-09-30", DECISIONS, basis=-0.06)
    assert estimate_basis(realized, DECISIONS, "2024-09-30", n_fixings=20) == pytest.approx(-0.06)


def test_realized_sofr_matters_for_a_month_already_in_progress():
    # Observed Mon Sep 23, 2024, after the Sep 18 cut (effective Sep 19).
    obs = pd.Timestamp("2024-09-23")
    contracts = contract_strip(obs)
    realized = make_synthetic_realized_sofr("2024-05-01", obs, DECISIONS, basis=BASIS)
    anchor = policy_level_on(obs, DECISIONS, after_announcement=False)  # 4.875
    expected = {"2024-11-07": -20.0, "2024-12-18": -15.0, "2025-01-29": -10.0}
    schedule = MEETINGS.append(pd.DatetimeIndex(["2026-01-28"]))
    prices = make_synthetic_market(obs, contracts, expected, anchor, BASIS, realized_fixings=realized)

    with_realized = fit_meeting_moves(obs, prices, schedule, anchor, BASIS, realized_fixings=realized)
    without = fit_meeting_moves(obs, prices, schedule, anchor, BASIS)

    assert with_realized.diagnostics["rmse_bps"] < 1e-3
    # Without published SOFR, Sep 1-18 (18 of 30 days) are priced at the new
    # rate instead of a rate 50 bp higher: the model rate is 50 * 18 / 30 = 30 bp
    # too low, so market minus model price is -30 bp.
    september = without.contracts.set_index("contract_id").loc["SR1-2024-09", "residual_bps"]
    assert september == pytest.approx(-30.0, abs=1e-3)
    assert without.diagnostics["rmse_bps"] > 5


# ---------------------------------------------------------------------------
# Many dates -> the table for #5
# ---------------------------------------------------------------------------


def synthetic_panel(obs_dates, decisions, expected):
    """A FAKE settlement panel (one row per contract per trade_date)."""
    frames = []
    for obs in pd.to_datetime(obs_dates):
        anchor = policy_level_on(obs, decisions, after_announcement=False)
        known = {}
        if (decisions["meeting_date"] == obs).any():
            post = policy_level_on(obs, decisions, after_announcement=True)
            known = {obs + pd.offsets.BDay(1): post - anchor}
        upcoming = {d: m for d, m in expected.items() if pd.Timestamp(d) > obs}
        day = make_synthetic_market(obs, contract_strip(obs), upcoming, anchor, BASIS, known_moves=known)
        frames.append(day.assign(trade_date=obs))
    return pd.concat(frames, ignore_index=True)


def test_fit_many_returns_the_table_issue5_expects():
    decisions = make_synthetic_decisions(["2024-06-12"], [-25])   # a cut announced on Jun 12
    expected = {"2024-07-31": -6.0, "2024-09-18": -18.0, "2024-11-07": -12.0}
    obs_dates = ["2024-06-11", "2024-06-12"]                      # the day before, and the decision day
    panel = synthetic_panel(obs_dates, decisions, {"2024-06-12": -25.0, **expected})

    implied, diagnostics = fit_many(panel, obs_dates, decisions, meeting_dates=MEETINGS, basis=BASIS)

    # Column names that PR #15 (Issue #5) reads.
    for column in ("obs_date", "meeting_date", "implied_cum_change_bps"):
        assert column in implied.columns
    assert len(diagnostics) == 2 and diagnostics["converged"].all()
    # Every price is matched on both days. On the decision day this needs the
    # anchor to be the OLD level until the cut takes effect on Jun 13.
    assert (diagnostics["rmse_bps"] < 1e-3).all()

    # Day before: the Jun 12 meeting is an unknown, and its -25 bp is recovered.
    before = implied[implied["obs_date"] == "2024-06-11"].set_index("meeting_date")
    assert before.loc["2024-06-12", "implied_move_bps"] == pytest.approx(-25.0, abs=1e-3)

    # Decision day: Jun 12 is known, so it is not an unknown, and the next
    # moves are still recovered because the known cut is applied to the path.
    on_day = implied[implied["obs_date"] == "2024-06-12"].set_index("meeting_date")
    assert pd.Timestamp("2024-06-12") not in on_day.index
    assert on_day.loc["2024-07-31", "implied_move_bps"] == pytest.approx(-6.0, abs=1e-3)
    assert on_day.loc["2024-09-18", "implied_cum_change_bps"] == pytest.approx(-24.0, abs=1e-3)


def test_fit_many_can_skip_dates_that_fail():
    decisions = make_synthetic_decisions(["2024-06-12"], [0])
    panel = synthetic_panel(["2024-06-03"], decisions, EXPECTED_BPS)
    dates = ["2024-06-03", "2024-06-04"]   # no prices on Jun 4

    with pytest.raises(ValueError, match="no usable contracts"):
        fit_many(panel, dates, decisions, meeting_dates=MEETINGS, basis=BASIS)

    implied, diagnostics = fit_many(panel, dates, decisions, meeting_dates=MEETINGS,
                                    basis=BASIS, skip_errors=True)
    assert diagnostics["error"].notna().sum() == 1
    assert set(implied["obs_date"]) == {pd.Timestamp("2024-06-03")}


# ---------------------------------------------------------------------------
# Bad input fails loudly
# ---------------------------------------------------------------------------


def test_synthetic_market_ignores_meetings_after_every_contract():
    far = {**EXPECTED_BPS, "2030-01-30": -100.0}   # far beyond the last contract
    pd.testing.assert_frame_equal(market(), make_synthetic_market(OBS, contract_strip(OBS), far, ANCHOR, BASIS))


def test_missing_columns_and_negative_ridge_are_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        fit_meeting_moves(OBS, market().drop(columns="settlement_price"), MEETINGS, ANCHOR)
    with pytest.raises(ValueError, match="ridge"):
        fit_meeting_moves(OBS, market(), MEETINGS, ANCHOR, ridge=-1)
