"""
Align SOFR futures reference periods with FOMC policy meetings.
"""

import pandas as pd

def align_fomc_with_futures(futures, fomc):
    """
    Align upcoming FOMC decisions by the date their policy takes effect.

    Parameters:
    - futures (pd.DataFrame): SOFR futures settlement observations
    - fomc (pd.DataFrame): FOMC meeting dates and realized policy decisions

    Returns (pd.DataFrame): Matched futures observations and FOMC meetings,
    including realized target midpoints and changes in basis points.
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

    for column in ("trade_date", "reference_start", "reference_end"):
        futures[column] = pd.to_datetime(futures[column])
    fomc["meeting_date"] = pd.to_datetime(fomc["meeting_date"])

    #handle FOMC inputs that do not explicitly include effective dates.
    if "effective_date" not in fomc.columns:
        fomc["effective_date"] = pd.NaT
    fomc["effective_date"] = pd.to_datetime(fomc["effective_date"])

    #holds have no rate-change effective date; use the next business day.
    #BDay accounts for weekends but not market holidays.
    missing_effective_date = fomc["effective_date"].isna()
    fomc.loc[missing_effective_date, "effective_date"] = (fomc.loc[missing_effective_date, "meeting_date"] + pd.offsets.BDay(1))

    if fomc["meeting_date"].duplicated().any():
        raise ValueError("Duplicate FOMC meeting dates")

    #only pre-announcement observations qualify as forecasts.
    #contract overlap is determined by policy effective date, not announcement date.
    fomc = fomc.sort_values("meeting_date").reset_index(drop = True)
    matched = []

    for _, meeting in fomc.iterrows():
        meeting_date = meeting["meeting_date"]
        effective_date = meeting["effective_date"]

        mask = ((futures["trade_date"] < meeting_date)
                & (futures["reference_start"] <= effective_date)
                & (futures["reference_end"] >= effective_date))

        subset = futures.loc[mask].copy()
        if subset.empty:
            continue

        for column in fomc.columns:
            subset[column] = meeting[column]

        matched.append(subset)

    if not matched:
        return pd.DataFrame(columns = [*futures.columns, *[column for column in fomc.columns if column not in futures.columns],
                                       "realized_target_midpoint", "realized_change_bps"])

    aligned = pd.concat(matched, ignore_index = True)

    aligned["realized_target_midpoint"] = (aligned["target_lower_after"] + aligned["target_upper_after"]) / 2

    aligned = aligned.rename(columns = {"change_bps": "realized_change_bps"})

    return aligned.sort_values(["trade_date", "contract_id", "meeting_date"]).reset_index(drop = True)
