"""
Validate and clean realized FOMC policy decisions.
"""

import pandas as pd


def validate_fomc_decisions(df):
    """
    Validate meeting-level realized FOMC policy decisions.

    Parameters:
    - df (pd.DataFrame): Meeting dates, target bounds, and decisions.

    Returns (pd.DataFrame): Validated decisions sorted by meeting date.

    Raises:ValueError: If the data fail a consistency check.
    """
    df = df.copy()

    #1. Check for duplicate meeting dates.
    if df["meeting_date"].duplicated().any():
        raise ValueError("Duplicate FOMC meeting dates found.")

    #2. Check for missing required values.
    required_columns = ["meeting_date",
                        "target_lower_before",
                        "target_upper_before",
                        "target_lower_after",
                        "target_upper_after",
                        "change_bps",
                        "decision"]

    if df[required_columns].isna().any().any():
        raise ValueError("Missing values in required FOMC fields.")

    #3. Check that lower bounds never exceed upper bounds.
    for suffix in ["before", "after"]:
        lower = df[f"target_lower_{suffix}"]
        upper = df[f"target_upper_{suffix}"]

        if (lower > upper).any():
            raise ValueError(f"Target lower bound exceeds upper bound ({suffix}).")

    #4. Verify that rate changes are calculated correctly (percentage points to basis points).
    midpoint_before = (df["target_lower_before"] + df["target_upper_before"]) / 2

    midpoint_after = (df["target_lower_after"] + df["target_upper_after"]) / 2

    expected_change = (midpoint_after - midpoint_before) * 100

    if not ((expected_change - df["change_bps"]).abs() < 1e-6).all():
        raise ValueError("Incorrect basis-point change detected.")

    #5. Verify that decision labels match the rate changes (positive, negative, and zero changes).
    expected_decision = df["change_bps"].apply(lambda x: "hike" if x > 0 else "cut" if x < 0 else "hold")

    if not (expected_decision == df["decision"]).all():
        raise ValueError("Decision labels do not match rate changes.")

    #6. Check that rate changes have effective dates.
    if "effective_date" in df.columns:
        changed = df["change_bps"] != 0
        if df.loc[changed, "effective_date"].isna().any():
            raise ValueError("Missing effective date for a policy rate change.")

    return df.sort_values("meeting_date").reset_index(drop = True)
