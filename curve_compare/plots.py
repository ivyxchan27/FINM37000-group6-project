"""Figures for the Issue #6 comparison (also used by #7's final visualizations)."""

import matplotlib.pyplot as plt

from curve_compare.compare import implied_paths_for_plot, summarize_by_days_to_meeting


def plot_sample_paths(world, obs_date, horizon_days=240,
                       title="SR1-implied Fed rate path: full model vs. simplified curve"):
    """One sample date: true step path vs. both methods' implied daily paths.

    Shows directly what the comparison is about: the full model's path is a
    step function that can only move on an FOMC date, so it tracks the true
    path exactly between meetings (noise aside); the simplified curve is
    smooth everywhere and blurs across each meeting date.
    """
    paths = implied_paths_for_plot(world, obs_date, horizon_days)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(paths.index, paths["true"], color="black", linewidth=2.0, label="True policy path")
    ax.plot(paths.index, paths["full_model"], linestyle="-", label="Full model (FOMC-aware)")
    ax.plot(paths.index, paths["simple_curve"], linestyle="--", label="Simplified curve")

    for meeting in world.meeting_dates:
        if paths.index[0] <= meeting <= paths.index[-1]:
            ax.axvline(meeting, color="grey", linewidth=0.6, alpha=0.5)

    ax.set_xlabel("Date")
    ax.set_ylabel("Implied rate (%)")
    ax.set_title(f"{title}\nobserved as of {obs_date.date()} (grey lines: FOMC meetings)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_error_by_days_to_meeting(results, width_days=7,
                                  title="Next-meeting forecast error vs. days to the meeting"):
    """MAE by weekly days-to-meeting bucket, for both methods (the #7 chart).

    Shows whether the simplified curve's extra error is concentrated near
    FOMC meetings, as the team README asks, or spread out evenly.
    """
    table = summarize_by_days_to_meeting(results, width_days)
    positions = range(len(table))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(positions, table["full_model_mae_bps"], marker="o", label="Full model MAE")
    ax.plot(positions, table["simple_curve_mae_bps"], marker="s", label="Simplified curve MAE")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(table.index.astype(str))
    ax.invert_xaxis()  # read left to right as the meeting gets closer
    ax.set_xlabel(f"Calendar days before the next meeting ({width_days}-day buckets)")
    ax.set_ylabel("MAE (bp)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
