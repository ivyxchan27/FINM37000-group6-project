"""
Construct the realized FOMC policy decision dataset.
"""

import pandas as pd
from fomc_data.fomc_calendar import fetch_fomc_calendar
from fomc_data.target_rates import fetch_target_rates


def build_fomc_decisions():
    """
    Match FOMC meetings to changes in the Fed target range.

    Returns (pd.DataFrame): Meeting-level target bounds, rate changes in basis points,
    decision labels, and inferred effective dates.
    """
    meetings = fetch_fomc_calendar()
    rates = fetch_target_rates()
    rows = []

    for meeting_date in meetings["meeting_date"]:
        #compare the latest target before the meeting with the first after it.
        before = rates.loc[rates["date"] < meeting_date].iloc[-1]
        after = rates.loc[rates["date"] > meeting_date].iloc[0]

        #verify whether the target range changes on the following day.
        #note:  this is an inferred date, not a statement-verified effective date.
        effective_date = after["date"]
        #record the effective date only when a change occurred.
        if (after["target_lower"] == before["target_lower"] and after["target_upper"] == before["target_upper"]):
            effective_date = pd.NaT

         #convert from percentage points to basis points
        change_bps = round((after["target_midpoint"] - before["target_midpoint"]) * 100, 2,)

        if change_bps > 0:
            decision = "hike"
        elif change_bps < 0:
            decision = "cut"
        else:
            decision = "hold"

        rows.append({"meeting_date": meeting_date,
                     "effective_date": effective_date,
                     "target_lower_before": before["target_lower"],
                     "target_upper_before": before["target_upper"],
                     "target_lower_after": after["target_lower"],
                     "target_upper_after": after["target_upper"],
                     "change_bps": change_bps,
                     "decision": decision})

    return pd.DataFrame(rows)
