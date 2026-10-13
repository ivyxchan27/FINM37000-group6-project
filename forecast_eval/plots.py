"""Forecast-error-by-horizon chart for the final figures (#7)."""

import matplotlib.pyplot as plt
import numpy as np

from forecast_eval.metrics import error_by_days_to_meeting, summarize_by_horizon, weekly_day_bins


def plot_error_by_horizon(errors, title="SOFR-implied Fed path: forecast error by horizon"):
    """Return a matplotlib Figure of forecast accuracy against meetings ahead.

    `errors` is the output of `build_forecast_errors`. The left panel shows the
    spread of errors at each horizon (box = middle 50%, whiskers = 5th to 95th
    percentile); the right panel shows MAE, RMSE and bias, all in bp.
    The caller decides whether to show or save it (fig.savefig(...)).
    """
    table = summarize_by_horizon(errors)
    horizons = table.index.to_numpy()

    fig, (ax_box, ax_stats) = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

    # Left: distribution of errors at each horizon.
    samples = [errors.loc[errors["horizon"] == h, "error_bps"].to_numpy() for h in horizons]
    ax_box.boxplot(samples, positions=horizons, whis=(5, 95), showfliers=False)
    ax_box.axhline(0, color="black", linewidth=0.8)
    ax_box.set_xlabel("Meetings ahead")
    ax_box.set_ylabel("Forecast error, implied - realized (bp)")
    ax_box.set_title("Error distribution")

    # Right: summary statistics at each horizon.
    ax_stats.plot(horizons, table["mae_bps"], marker="o", label="MAE")
    ax_stats.plot(horizons, table["rmse_bps"], marker="s", label="RMSE")
    ax_stats.plot(horizons, table["bias_bps"], marker="^", linestyle="--", label="Bias (mean error)")
    ax_stats.axhline(0, color="black", linewidth=0.8)
    ax_stats.set_xlabel("Meetings ahead")
    ax_stats.set_ylabel("bp")
    ax_stats.set_title("Accuracy statistics")
    ax_stats.set_xticks(horizons)
    ax_stats.legend()

    for ax in (ax_box, ax_stats):
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_error_by_days_to_meeting(errors, horizon=1, width_days=7,
                                  title="SOFR-implied Fed path: error by days before the meeting"):
    """Return a matplotlib Figure of MAE, RMSE and bias against days before the meeting.

    Uses evenly spaced buckets of `width_days` days (weekly by default) at one
    horizon, so the x-axis reads as "weeks before the meeting". Empty buckets
    are left out. The caller decides whether to show or save it.
    """
    bins = weekly_day_bins(errors, horizon=horizon, width_days=width_days)
    table = error_by_days_to_meeting(errors, horizon=horizon, bins=bins, open_ended=False)
    positions = np.arange(len(table))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(positions, table["mae_bps"], marker="o", label="MAE")
    ax.plot(positions, table["rmse_bps"], marker="s", label="RMSE")
    ax.plot(positions, table["bias_bps"], marker="^", linestyle="--", label="Bias (mean error)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(table.index.astype(str))
    ax.invert_xaxis()  # read left to right as the meeting gets closer
    ax.set_xlabel(f"Calendar days before the meeting ({width_days}-day buckets, horizon {horizon})")
    ax.set_ylabel("bp")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig
