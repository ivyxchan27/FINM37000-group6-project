"""Unit tests for the SR1 / SR3 pricing code (Issue #3).

How to run (from the repository root):
    python -m pytest tests/ -v

Every expected number below is worked out BY HAND (see the comment above each
test) or by a separate reference calculation. We never get the expected value
by calling the function we are testing.

Units: rates are in percent (5.00 = 5%), prices are index points.
"""

import pandas as pd
import pytest

from sofr_pricing import (
    contract_price,
    fixing_dates,
    get_rate_blocks,
    price_contracts,
    sr1_price,
    sr1_rate,
    sr3_price,
    sr3_rate,
)


# ---------------------------------------------------------------------------
# Small helpers used by the tests
# ---------------------------------------------------------------------------


def assert_close(actual, expected, tolerance=1e-9):
    """Check two numbers are equal up to a tiny rounding error."""
    assert abs(actual - expected) < tolerance, f"{actual} is not close to {expected}"


def make_path(start, end, rate, moves=None, holidays=()):
    """Build a SOFR fixing path: `rate` everywhere, plus optional rate changes.

    `moves` is a dictionary {effective_date: change_in_percent}.
    A change applies from its effective date onwards.
    Example: {"2024-06-20": -0.25} is a 25 bp cut starting on June 20.
    """
    dates = fixing_dates(start, end, holidays)
    path = pd.Series(float(rate), index=dates)

    if moves is not None:
        for effective_date, change in moves.items():
            path[path.index >= pd.Timestamp(effective_date)] += change

    return path


def reference_sr3_rate(path, start, end):
    """A second, independent way to compute the SR3 rate (used as a cross-check).

    The code under test walks through the period day by day. This reference
    looks at each FIXING instead and asks: "how many days of the period does
    this fixing cover?"
    """
    start = pd.Timestamp(start)
    end_exclusive = pd.Timestamp(end) + pd.Timedelta(days=1)
    total_days = (end_exclusive - start).days

    dates = list(path.index)
    growth_factor = 1.0
    for i in range(len(dates)):
        # This fixing is in force from its own date until the next fixing date.
        if i + 1 < len(dates):
            in_force_until = dates[i + 1]
        else:
            in_force_until = end_exclusive

        first_day = max(dates[i], start)
        last_day_exclusive = min(in_force_until, end_exclusive)
        days = (last_day_exclusive - first_day).days

        if days > 0:
            growth_factor *= 1 + days * path.iloc[i] / 100 / 360

    return (growth_factor - 1) * 360 / total_days * 100


# ---------------------------------------------------------------------------
# get_rate_blocks: how a period is cut into pieces
# ---------------------------------------------------------------------------


def test_blocks_give_friday_three_days():
    # Wed Jun 19 to Tue Jun 25, 2024. Give every fixing a different rate so we
    # can see which one is used on which day.
    dates = fixing_dates("2024-06-19", "2024-06-25")
    path = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates)  # Wed Thu Fri Mon Tue

    blocks = get_rate_blocks(path, "2024-06-19", "2024-06-25")

    # Friday's rate (3.0) also covers Saturday and Sunday, so it gets 3 days.
    assert blocks == [[1.0, 1], [2.0, 1], [3.0, 3], [4.0, 1], [5.0, 1]]


def test_blocks_start_on_a_weekend_uses_friday_rate():
    # Saturday Jun 1, 2024: the rate in force comes from Friday May 31.
    dates = fixing_dates("2024-05-30", "2024-06-04")
    path = pd.Series([1.0, 2.0, 3.0, 4.0], index=dates)  # Thu May 30, Fri May 31, Mon Jun 3, Tue Jun 4

    blocks = get_rate_blocks(path, "2024-06-01", "2024-06-04")

    # Sat + Sun use Friday's 2.0 (2 days), then Mon 3.0 and Tue 4.0.
    assert blocks == [[2.0, 2], [3.0, 1], [4.0, 1]]


def test_fixing_dates_skips_weekends_and_holidays():
    dates = fixing_dates("2024-12-23", "2024-12-27", holidays=["2024-12-25"])
    names = [d.strftime("%a") for d in dates]
    assert names == ["Mon", "Tue", "Thu", "Fri"]  # no Wednesday (Christmas)


# ---------------------------------------------------------------------------
# SR1: average of daily SOFR over the calendar month
# ---------------------------------------------------------------------------


def test_sr1_flat_path_gives_that_rate():
    path = make_path("2024-05-01", "2025-04-30", 5.31)
    assert_close(sr1_rate(path, "2024-06-01", "2024-06-30"), 5.31)


def test_sr1_price_is_100_minus_rate():
    path = make_path("2024-05-01", "2025-04-30", 5.31)
    assert_close(sr1_price(path, "2024-06-01", "2024-06-30"), 94.69)


def test_sr1_midweek_cut():
    # June 2024 has 30 days. A 25 bp cut takes effect on Thu Jun 20.
    #   Jun 1-19 : 19 days at 5.00
    #   Jun 20-30: 11 days at 4.75
    # average = (19 * 5.00 + 11 * 4.75) / 30 = 147.25 / 30
    path = make_path("2024-05-01", "2024-07-31", 5.00, {"2024-06-20": -0.25})
    assert_close(sr1_rate(path, "2024-06-01", "2024-06-30"), 147.25 / 30)


def test_sr1_monday_cut_keeps_old_rate_over_the_weekend():
    # The cut takes effect on Mon Jun 17. Saturday Jun 15 and Sunday Jun 16
    # still use Friday's old rate.
    #   Jun 1-16 : 16 days at 5.00
    #   Jun 17-30: 14 days at 4.75
    # average = (16 * 5.00 + 14 * 4.75) / 30 = 146.5 / 30
    path = make_path("2024-05-01", "2024-07-31", 5.00, {"2024-06-17": -0.25})
    assert_close(sr1_rate(path, "2024-06-01", "2024-06-30"), 146.5 / 30)


def test_sr1_month_that_starts_on_a_weekend():
    # June 1, 2024 is a Saturday. Make Friday May 31 special (6.0) and every
    # fixing from Monday Jun 3 on equal to 4.0; everything else is 5.0.
    #   Jun 1-2  : 2 days at 6.0 (carried from Friday)
    #   Jun 3-30 : 28 days at 4.0
    # average = (2 * 6 + 28 * 4) / 30 = 124 / 30
    dates = fixing_dates("2024-05-01", "2024-06-30")
    path = pd.Series(5.0, index=dates)
    path["2024-05-31"] = 6.0
    path[path.index >= pd.Timestamp("2024-06-03")] = 4.0

    assert_close(sr1_rate(path, "2024-06-01", "2024-06-30"), 124 / 30)


def test_sr1_moves_by_days_remaining_over_days_in_month():
    # A +0.10 move that starts with 11 of the 30 days left should raise the
    # monthly average by 0.10 * 11 / 30.
    base = make_path("2024-05-01", "2024-07-31", 5.00)
    bumped = make_path("2024-05-01", "2024-07-31", 5.00, {"2024-06-20": 0.10})

    difference = sr1_rate(bumped, "2024-06-01", "2024-06-30") - sr1_rate(
        base, "2024-06-01", "2024-06-30"
    )
    assert_close(difference, 0.10 * 11 / 30)


# ---------------------------------------------------------------------------
# SR3: SOFR compounded over the reference quarter
# ---------------------------------------------------------------------------


def test_sr3_flat_path_matches_closed_form():
    # Dec 18, 2024 (Wed) to Mar 18, 2025 (Tue) is 91 days = 13 whole weeks.
    # With no holidays, one Wed-to-Tue week has these fixings and day counts:
    #   Wed 1 day, Thu 1 day, Fri 3 days (Fri+Sat+Sun), Mon 1 day, Tue 1 day.
    # With x = rate / 36000, one week multiplies the money by
    #   (1 + x) ** 4 * (1 + 3 * x)
    # and 13 weeks give A = that number ** 13.
    rate = 5.00
    x = rate / 36000
    growth = ((1 + x) ** 4 * (1 + 3 * x)) ** 13
    expected = (growth - 1) * 360 / 91 * 100

    path = make_path("2024-12-01", "2025-03-31", rate)
    assert_close(sr3_rate(path, "2024-12-18", "2025-03-18"), expected)


def test_sr3_is_a_bit_above_the_simple_average_because_of_compounding():
    # Interest earns interest, so the compounded rate is higher than the plain
    # average. Rule of thumb: about rate * rate * days / (2 * 360 * 100).
    # At 5% over 91 days that is about 0.03 percentage points (3 basis points).
    path = make_path("2024-05-01", "2025-04-30", 5.00)

    compounded = sr3_rate(path, "2024-12-18", "2025-03-18")
    simple_average = sr1_rate(path, "2024-12-18", "2025-03-18")

    assert compounded > simple_average
    assert abs((compounded - simple_average) - 0.0314) < 0.002


def test_sr3_with_cuts_and_holidays_matches_the_reference_calculation():
    holidays = ["2024-12-25", "2025-01-01", "2025-01-20", "2025-02-17"]
    moves = {"2024-12-19": -0.25, "2025-01-30": -0.25, "2025-03-20": 0.10}
    path = make_path("2024-12-01", "2025-03-31", 4.50, moves, holidays)

    expected = reference_sr3_rate(path, "2024-12-18", "2025-03-18")
    assert_close(sr3_rate(path, "2024-12-18", "2025-03-18"), expected)


def test_sr3_zero_rates_give_price_100():
    path = make_path("2024-05-01", "2025-04-30", 0.0)
    assert_close(sr3_price(path, "2024-12-18", "2025-03-18"), 100.0)


def test_sr3_moves_by_about_the_share_of_days_remaining():
    # A +0.10 move effective Feb 3, 2025 is in force for the last 44 of the 91
    # days (Feb 3 to Mar 18). The SR3 rate should rise by about 0.10 * 44 / 91.
    # "About" because compounding adds a tiny extra effect.
    base = make_path("2024-12-01", "2025-03-31", 5.00)
    bumped = make_path("2024-12-01", "2025-03-31", 5.00, {"2025-02-03": 0.10})

    difference = sr3_rate(bumped, "2024-12-18", "2025-03-18") - sr3_rate(
        base, "2024-12-18", "2025-03-18"
    )
    expected = 0.10 * 44 / 91
    assert abs(difference - expected) / expected < 0.02  # within 2%


# ---------------------------------------------------------------------------
# A path that mixes realized (past) and model (future) rates
# ---------------------------------------------------------------------------


def test_path_mixing_realized_and_model_days():
    # Imagine today is Mon Jun 17. SOFR up to Fri Jun 14 is already known
    # (realized, 5.30). After that we use a model guess (5.00).
    #   Jun 1-16 : 16 days at 5.30
    #   Jun 17-30: 14 days at 5.00
    # average = (16 * 5.30 + 14 * 5.00) / 30
    dates = fixing_dates("2024-05-01", "2024-07-31")
    path = pd.Series(5.00, index=dates)
    path[path.index <= pd.Timestamp("2024-06-14")] = 5.30

    assert_close(sr1_rate(path, "2024-06-01", "2024-06-30"), (16 * 5.30 + 14 * 5.00) / 30)


# ---------------------------------------------------------------------------
# Choosing the right contract and pricing a whole table
# ---------------------------------------------------------------------------


def test_contract_price_picks_the_right_formula():
    path = make_path("2024-05-01", "2025-04-30", 5.00, {"2024-06-20": -0.25})

    sr1 = contract_price("SR1", path, "2024-06-01", "2024-06-30")
    sr3 = contract_price("SR3", path, "2024-06-19", "2024-09-17")

    assert sr1 == sr1_price(path, "2024-06-01", "2024-06-30")
    assert sr3 == sr3_price(path, "2024-06-19", "2024-09-17")


def test_contract_price_rejects_unknown_product():
    path = make_path("2024-05-01", "2025-04-30", 5.00)
    with pytest.raises(ValueError, match="unknown product root"):
        contract_price("ZN", path, "2024-06-01", "2024-06-30")


def test_price_contracts_prices_each_row_and_keeps_the_index():
    path = make_path("2024-05-01", "2025-04-30", 5.00, {"2024-06-20": -0.25})
    table = pd.DataFrame(
        {
            "root": ["SR1", "SR3"],
            "reference_start": pd.to_datetime(["2024-06-01", "2024-06-19"]),
            "reference_end": pd.to_datetime(["2024-06-30", "2024-09-17"]),
        },
        index=[10, 20],  # unusual index on purpose: it must be preserved
    )

    prices = price_contracts(path, table)

    assert list(prices.index) == [10, 20]
    assert_close(prices[10], sr1_price(path, "2024-06-01", "2024-06-30"))
    assert_close(prices[20], sr3_price(path, "2024-06-19", "2024-09-17"))


# ---------------------------------------------------------------------------
# Bad input must fail loudly instead of giving a wrong price
# ---------------------------------------------------------------------------


def test_error_when_fixings_start_after_the_period():
    path = make_path("2024-06-03", "2024-07-31", 5.0)  # starts Mon Jun 3
    with pytest.raises(ValueError, match="after the period start"):
        sr1_rate(path, "2024-06-01", "2024-06-30")


def test_error_when_fixings_end_before_the_period_does():
    path = make_path("2024-05-01", "2024-06-14", 5.0)  # stops Fri Jun 14
    with pytest.raises(ValueError, match="too early to cover"):
        sr1_rate(path, "2024-06-01", "2024-06-30")


def test_long_weekend_at_the_end_is_fine_but_a_gap_in_the_middle_is_not():
    # Fixings up to Fri Jun 28 are enough: Saturday and Sunday reuse Friday.
    ok_path = make_path("2024-05-01", "2024-06-28", 5.0)
    assert_close(sr1_rate(ok_path, "2024-06-01", "2024-06-30"), 5.0)

    # Remove ten fixings from the middle: that is missing data, not a weekend.
    dates = fixing_dates("2024-05-01", "2024-07-31")
    dates_with_hole = dates.delete(range(20, 30))
    path_with_hole = pd.Series(5.0, index=dates_with_hole)
    with pytest.raises(ValueError, match="gap between consecutive fixings"):
        sr1_rate(path_with_hole, "2024-06-01", "2024-06-30")


def test_error_for_missing_values_unsorted_dates_or_wrong_index():
    path = make_path("2024-05-01", "2024-07-31", 5.0)

    with_missing_value = path.copy()
    with_missing_value.iloc[5] = float("nan")
    with pytest.raises(ValueError, match="missing values"):
        sr1_rate(with_missing_value, "2024-06-01", "2024-06-30")

    with pytest.raises(ValueError, match="sorted and unique"):
        sr1_rate(path.iloc[::-1], "2024-06-01", "2024-06-30")  # reversed order

    with pytest.raises(TypeError):
        sr1_rate(pd.Series([5.0, 5.0]), "2024-06-01", "2024-06-30")  # no dates as index


def test_error_when_period_ends_before_it_starts():
    path = make_path("2024-05-01", "2025-04-30", 5.0)
    with pytest.raises(ValueError, match="before start"):
        sr1_rate(path, "2024-06-30", "2024-06-01")
