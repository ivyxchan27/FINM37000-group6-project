"""SOFR futures pricing framework (Issue #3).

Turns an assumed path of daily SOFR rates into theoretical SR1 and SR3
futures prices.
"""

from sofr_pricing.fixing_calendar import MAX_CARRY_DAYS, fixing_dates
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
    "MAX_CARRY_DAYS",
    "contract_price",
    "fixing_dates",
    "get_rate_blocks",
    "price_contracts",
    "sr1_price",
    "sr1_rate",
    "sr3_price",
    "sr3_rate",
]
