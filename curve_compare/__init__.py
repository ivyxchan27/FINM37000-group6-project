"""Compare the full FOMC-aware model with a simplified SOFR curve (Issue #6).

See CURVE_COMPARE.md for definitions, scope, and sources.
"""

from curve_compare.compare import (
    compare_at_date,
    default_obs_dates,
    implied_paths_for_plot,
    method_difference_bps,
    next_meeting_after,
    run_comparison,
    sr1_strip_as_of,
    summarize_by_days_to_meeting,
    summarize_by_regime,
    summarize_errors,
    summarize_errors_excluding_before_first_meeting,
)
from curve_compare.full_model import anchor_sofr_rate, implied_daily_path, recover_full_model_path
from curve_compare.plots import plot_error_by_days_to_meeting, plot_sample_paths
from curve_compare.simple_curve import build_simple_zero_curve, simple_curve_daily_path, simple_curve_path
from curve_compare.synthetic import SyntheticWorld, generate_synthetic_world

__all__ = [
    "SyntheticWorld",
    "generate_synthetic_world",
    "anchor_sofr_rate",
    "recover_full_model_path",
    "implied_daily_path",
    "build_simple_zero_curve",
    "simple_curve_daily_path",
    "simple_curve_path",
    "sr1_strip_as_of",
    "next_meeting_after",
    "default_obs_dates",
    "compare_at_date",
    "run_comparison",
    "summarize_errors",
    "summarize_errors_excluding_before_first_meeting",
    "method_difference_bps",
    "summarize_by_regime",
    "summarize_by_days_to_meeting",
    "implied_paths_for_plot",
    "plot_sample_paths",
    "plot_error_by_days_to_meeting",
]
