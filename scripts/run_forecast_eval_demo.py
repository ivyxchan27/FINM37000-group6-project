"""Demo of the Issue #5 evaluation on SYNTHETIC data (no real prices yet).

Run from the repository root:
    python -m scripts.run_forecast_eval_demo
    python -m scripts.run_forecast_eval_demo --plot forecast_error_by_horizon.png

The fake implied paths are the true path plus a bias and noise that both grow
with the horizon, so the tables should show bias and error rising with
meetings ahead. The synthetic noise does not depend on days to the meeting,
so the days-before-meeting table is flat here; real data should not be.
"""

import argparse

import pandas as pd

from forecast_eval import (
    build_forecast_errors,
    error_by_days_to_meeting,
    hit_rate_sensitivity,
    make_synthetic_decisions,
    make_synthetic_implied_paths,
    summarize_by_horizon,
    summarize_by_regime,
)

# Injected bias and noise (bp) at horizons 1..6. Known inputs, so we can check the output.
BIAS_BPS = [1, 2, 4, 6, 8, 10]
NOISE_BPS = [3, 6, 9, 12, 15, 18]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot", metavar="PATH", help="also save the forecast-error-by-horizon chart here")
    args = parser.parse_args()

    realized = make_synthetic_decisions()
    implied = make_synthetic_implied_paths(realized, bias_bps=BIAS_BPS, noise_bps=NOISE_BPS, seed=37000)
    errors = build_forecast_errors(implied, realized)

    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.2f}".format)
    print("SYNTHETIC DATA - placeholder until #4 produces real implied paths\n")
    print(f"Injected bias by horizon (bp): {BIAS_BPS}")
    print(f"Injected noise s.d. by horizon (bp): {NOISE_BPS}\n")
    print("Accuracy by horizon (meetings ahead):")
    print(summarize_by_horizon(errors), "\n")
    print("Direction hit rate at different hike / hold / cut thresholds:")
    print(hit_rate_sensitivity(errors), "\n")
    print("Horizon 1 accuracy by calendar days before the meeting:")
    print(error_by_days_to_meeting(errors, horizon=1), "\n")
    print("Accuracy by regime and horizon:")
    print(summarize_by_regime(errors))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")  # write to a file without opening a window
        from forecast_eval import plot_error_by_horizon
        from forecast_eval import plot_error_by_days_to_meeting
        plot_error_by_horizon(errors).savefig(args.plot, dpi=150)
        weekly_path = args.plot.rsplit(".", 1)[0] + "_by_week.png"
        plot_error_by_days_to_meeting(errors).savefig(weekly_path, dpi=150)
        print(f"\nSaved charts to {args.plot} and {weekly_path}")


if __name__ == "__main__":
    main()
