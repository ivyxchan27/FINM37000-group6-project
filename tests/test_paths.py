"""Tests for building SOFR paths and the calendar helpers (Issue #4).

Run from the repository root:  python -m pytest tests/ -v
"""

import pandas as pd
import pytest

from sofr_pricing import (
    effective_date,
    fixing_dates,
    last_known_fixing_date,
    splice_realized,
    step_fixings,
)


def test_step_fixings_applies_each_move_from_its_effective_date():
    dates = fixing_dates("2024-06-17", "2024-06-21")  # Mon to Fri
    path = step_fixings(dates, 5.00, {"2024-06-19": -0.25, "2024-06-21": -0.25}, basis=0.05)
    # Mon, Tue: 5.05 | Wed, Thu: 4.80 | Fri: 4.55
    assert path.tolist() == pytest.approx([5.05, 5.05, 4.80, 4.80, 4.55])


def test_effective_date_is_the_next_fixing_date_and_skips_holidays():
    dates = fixing_dates("2024-12-16", "2024-12-31", holidays=["2024-12-25"])
    assert effective_date("2024-12-18", dates) == pd.Timestamp("2024-12-19")
    # Announced Tue Dec 24; Wed Dec 25 is a holiday, so it starts Thu Dec 26.
    assert effective_date("2024-12-24", dates) == pd.Timestamp("2024-12-26")
    with pytest.raises(ValueError, match="no fixing date after"):
        effective_date("2024-12-31", dates)


def test_last_known_fixing_date_is_the_previous_business_day():
    dates = fixing_dates("2024-06-10", "2024-06-21")
    assert last_known_fixing_date("2024-06-17", dates) == pd.Timestamp("2024-06-14")  # Mon -> Fri
    assert last_known_fixing_date("2024-06-14", dates) == pd.Timestamp("2024-06-13")
    with pytest.raises(ValueError, match="no fixing date before"):
        last_known_fixing_date("2024-06-10", dates)


def test_splice_uses_realized_up_to_the_cutoff_and_model_after():
    dates = fixing_dates("2024-06-10", "2024-06-21")
    realized = pd.Series(5.30, index=dates)
    model = pd.Series(5.00, index=dates)

    spliced = splice_realized(realized, model, "2024-06-14")

    assert (spliced[spliced.index <= "2024-06-14"] == 5.30).all()
    assert (spliced[spliced.index >= "2024-06-17"] == 5.00).all()
    assert spliced.index.is_unique and spliced.index.is_monotonic_increasing


def test_splice_rejects_a_cutoff_before_any_realized_fixing():
    dates = fixing_dates("2024-06-10", "2024-06-21")
    series = pd.Series(5.0, index=dates)
    with pytest.raises(ValueError, match="no realized fixings"):
        splice_realized(series, series, "2024-06-01")
