"""SOFR futures pricing framework (Issue #3) and Fed-path estimation (Issue #4).

#3: turns an assumed path of daily SOFR rates into theoretical SR1 and SR3
    futures prices.
#4: runs that in reverse: finds the rate move at each upcoming FOMC meeting
    that best explains the observed SR1 and SR3 prices.
"""

from sofr_pricing.contract_dates import (
    contract_strip,
    sr1_reference_period,
    sr3_reference_period,
    third_wednesday,
)
from sofr_pricing.estimation import (
    DEFAULT_RIDGE,
    FitResult,
    estimate_basis,
    fit_many,
    fit_meeting_moves,
    policy_level_on,
    upcoming_meetings,
)
from sofr_pricing.fixing_calendar import (
    MAX_CARRY_DAYS,
    effective_date,
    fixing_dates,
    last_known_fixing_date,
)
from sofr_pricing.paths import splice_realized, step_fixings
from sofr_pricing.pricing import (
    contract_price,
    get_rate_blocks,
    price_contracts,
    sr1_price,
    sr1_rate,
    sr3_price,
    sr3_rate,
)

__all__ = [
    "DEFAULT_RIDGE",
    "FitResult",
    "MAX_CARRY_DAYS",
    "contract_price",
    "contract_strip",
    "effective_date",
    "estimate_basis",
    "fit_many",
    "fit_meeting_moves",
    "fixing_dates",
    "get_rate_blocks",
    "last_known_fixing_date",
    "policy_level_on",
    "price_contracts",
    "splice_realized",
    "sr1_price",
    "sr1_rate",
    "sr1_reference_period",
    "sr3_price",
    "sr3_rate",
    "sr3_reference_period",
    "step_fixings",
    "third_wednesday",
    "upcoming_meetings",
]
