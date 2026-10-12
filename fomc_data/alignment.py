"""
Align SOFR futures reference periods with FOMC policy meetings.
"""

import pandas as pd

def align_fomc_with_futures(futures, fomc):
    """
    Aligns SOFR futures observations with upcoming FOMC meetings that fall
    within each contract's reference period.

    Parameters:
    - futures (pd.DataFrame): contains SOFR futures settlement observations
    - fomc (pd.DataFrame): contains FOMC meeting dates and realized rate decisions

    Returns (pd.DataFrame): with matched futures observations and FOMC meetings,
    including realized policy rate changes and target midpoints
    """
    required_futures = {"trade_date",
                        "contract_id",
                        "reference_start",
                        "reference_end"}
    required_fomc = {"meeting_date",
                     "target_lower_after",
                     "target_upper_after",
                     "change_bps",
                     "decision"}

    if missing := required_futures - set(futures.columns):
        raise ValueError(f"Missing futures columns: {sorted(missing)}")

    if missing := required_fomc - set(fomc.columns):
        raise ValueError(f"Missing FOMC columns: {sorted(missing)}")

    futures = futures.copy()
    fomc = fomc.copy()

    for col in ("trade_date", "reference_start", "reference_end"):
        futures[col] = pd.to_datetime(futures[col])
    fomc["meeting_date"] = pd.to_datetime(fomc["meeting_date"])

    if fomc["meeting_date"].duplicated().any():
        raise ValueError("Duplicate FOMC meeting dates")
    #sort meetings chronologically.
    fomc = fomc.sort_values("meeting_date").reset_index(drop=True)

    #match each futures observation to relevant meetings.
    matched = []

    for _, meeting in fomc.iterrows():
        date = meeting["meeting_date"]

        mask = ((futures["trade_date"] < date)
                & (futures["reference_start"] <= date)
                & (futures["reference_end"] >= date))

        subset = futures.loc[mask].copy()

        if subset.empty:
            continue

        for col in fomc.columns:
            subset[col] = meeting[col]

        matched.append(subset)

    if not matched:
        return pd.DataFrame(columns = [*futures.columns, *[col for col in fomc.columns if col not in futures.columns],
                                       "realized_target_midpoint", "realized_change_bps"])

    aligned = pd.concat(matched, ignore_index=True)

    aligned["realized_target_midpoint"] = (aligned["target_lower_after"] + aligned["target_upper_after"]) / 2

    aligned = aligned.rename(columns = {"change_bps": "realized_change_bps"})

    return aligned.sort_values(["trade_date", "contract_id", "meeting_date"]).reset_index(drop = True)
