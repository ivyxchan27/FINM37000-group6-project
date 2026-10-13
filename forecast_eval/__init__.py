"""Fed-path forecast accuracy evaluation (Issue #5).

Scores how well SOFR-futures-implied Fed paths (from #4) predicted the
realized FOMC decisions (from #2), at horizons of 1 to 6 meetings ahead.
See FORECAST_EVAL.md for the definitions and the input tables.
"""

from forecast_eval.inputs import validate_implied_paths, validate_realized_decisions
from forecast_eval.metrics import (
    DEFAULT_DAY_BINS,
    DEFAULT_REGIMES,
    DIRECTION_THRESHOLD_BPS,
    MAX_HORIZON,
    SENSITIVITY_THRESHOLDS_BPS,
    build_forecast_errors,
    classify_direction,
    error_by_days_to_meeting,
    hit_rate_sensitivity,
    summarize_by_horizon,
    summarize_by_regime,
)
from forecast_eval.plots import plot_error_by_horizon
from forecast_eval.synthetic import make_synthetic_decisions, make_synthetic_implied_paths

__all__ = [
    "DEFAULT_DAY_BINS",
    "DEFAULT_REGIMES",
    "DIRECTION_THRESHOLD_BPS",
    "MAX_HORIZON",
    "SENSITIVITY_THRESHOLDS_BPS",
    "build_forecast_errors",
    "classify_direction",
    "error_by_days_to_meeting",
    "hit_rate_sensitivity",
    "make_synthetic_decisions",
    "make_synthetic_implied_paths",
    "plot_error_by_horizon",
    "summarize_by_horizon",
    "summarize_by_regime",
    "validate_implied_paths",
    "validate_realized_decisions",
]
