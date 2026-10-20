"""
Validate FOMC alignment against actual SOFR futures settlement data.
"""

import pandas as pd
import pytest

from fomc_data.alignment import align_fomc_with_futures


def test_fomc_sofr_integration():
    """
    Check that real futures settlements align with realized FOMC decisions.
    """
    futures_path = "data/processed/sofr_futures_settlements.parquet"
    fomc_path = "data/processed/fomc_meetings.csv"

    #real-data integration requires both processed input datasets.
    from pathlib import Path

    if not Path(futures_path).exists() or not Path(fomc_path).exists():
        pytest.skip("Processed futures or FOMC data not available")

    futures = pd.read_parquet(futures_path)
    fomc = pd.read_csv(fomc_path)

    aligned = align_fomc_with_futures(futures, fomc)

    assert not aligned.empty

    #all source FOMC meetings should be represented.
    assert set(aligned["meeting_date"]) == set(pd.to_datetime(fomc["meeting_date"]))

    #normalize dates before validating.
    date_cols = ["trade_date", "meeting_date", "effective_date", "reference_start", "reference_end"]
    for col in date_cols:
        aligned[col] = pd.to_datetime(aligned[col])

    #futures observations must precede FOMC announcements.
    assert (aligned["trade_date"] < aligned["meeting_date"]).all()

    #policy effective dates must fall within contract reference periods.
    assert ((aligned["effective_date"] >= aligned["reference_start"]) & (aligned["effective_date"] <= aligned["reference_end"])).all()

    #no duplicate trade/contract/meeting combinations.
    assert not aligned.duplicated(["trade_date", "contract_id", "meeting_date"]).any()

    #all effective dates must be populated.
    assert aligned["effective_date"].notna().all()

    #realized rate changes must match the source FOMC decisions.
    expected = fomc.set_index("meeting_date")["change_bps"]
    actual = aligned["meeting_date"].map(expected.rename_axis("meeting_date").set_axis(pd.to_datetime(expected.index)))

    assert actual.notna().all()
    assert (aligned["realized_change_bps"] == actual).all()

    #both futures products should appear in the aligned dataset.
    assert {"SR1", "SR3"}.issubset(set(aligned["root"]))