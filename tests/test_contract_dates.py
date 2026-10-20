"""Tests for IMM dates and contract reference periods (Issue #4).

The expected dates are plain calendar facts (for example, Jan 1, 2024 was a
Monday, so its Wednesdays are Jan 3, 10 and 17).
"""

import pandas as pd
import pytest

from sofr_pricing import contract_strip, sr1_reference_period, sr3_reference_period, third_wednesday


@pytest.mark.parametrize(
    ("year", "month", "expected"),
    [(2024, 1, "2024-01-17"), (2024, 3, "2024-03-20"), (2024, 12, "2024-12-18"),
     (2025, 3, "2025-03-19"), (2026, 9, "2026-09-16")],
)
def test_third_wednesday(year, month, expected):
    result = third_wednesday(year, month)
    assert result == pd.Timestamp(expected)
    assert result.day_name() == "Wednesday"


def test_sr1_period_is_the_calendar_month():
    assert sr1_reference_period(2024, 2) == (pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-29"))


def test_sr3_period_ends_the_day_before_the_next_imm_date():
    start, end = sr3_reference_period(2024, 12)
    assert start == pd.Timestamp("2024-12-18")
    assert end == pd.Timestamp("2025-03-18")  # Mar 19, 2025 is the next IMM date


def test_sr3_quarters_join_without_gaps():
    periods = [sr3_reference_period(2025, m) for m in (3, 6, 9, 12)]
    for (_, end), (next_start, _) in zip(periods, periods[1:]):
        assert next_start == end + pd.Timedelta(days=1)


def test_sr3_needs_a_quarterly_month():
    with pytest.raises(ValueError, match="SR3 contract months"):
        sr3_reference_period(2025, 4)


def test_contract_strip_keeps_only_contracts_still_running():
    obs = pd.Timestamp("2024-06-03")
    strip = contract_strip(obs, n_sr1=13, n_sr3=6)

    assert (strip["root"] == "SR1").sum() == 13
    assert (strip["root"] == "SR3").sum() == 6
    assert (strip["reference_end"] >= obs).all()
    # The SR3 quarter in progress on Jun 3, 2024 began on Mar 20, 2024.
    first_sr3 = strip[strip["root"] == "SR3"].iloc[0]
    assert first_sr3["reference_start"] == pd.Timestamp("2024-03-20")
    assert first_sr3["contract_id"] == "SR3-2024-03"
