# Full Model vs. Simplified SOFR Curve (Issue #6)

The `curve_compare` package asks: does explicitly modeling FOMC meeting timing and
SR1 settlement conventions give a more accurate market-implied Fed rate path than a
generic smooth curve that ignores FOMC timing?

**Status:** the comparison runs on **synthetic data**. It generates a known, seeded
policy path and SR1 prices, then recovers the path from those prices. The repository
contains data collection, pricing, FOMC calendar, and forecast evaluation modules, but
this comparison does not import them. Real settlements and fitted implied paths are
not inputs to this demo; see "Pending" for possible integration work.

## Scope

Issue #6 covers a large space (SR1 vs. SR3, several forecast horizons, real vs. synthetic
data). Following the professor's check-in feedback to focus on a single contract and a
single horizon, this comparison is scoped to:

- **SR1 only** (one-month SOFR futures), not SR3.
- **Next-meeting horizon only** (horizon 1 in `forecast_eval`'s terms), not horizons 2-6.
- **Synthetic data** (a seeded, fully-known policy path), not real settlements.

SR3 and multi-horizon comparisons are natural follow-ups once this scope is validated; see
"Pending" below. They are not implemented here.

## The two methods

Both methods use one date's strip of SR1 settlement prices to estimate a daily
SOFR rate path. The full model also uses meeting dates, a recent SOFR anchor
restricted to fixings since the latest meeting, and all observed fixings in
the elapsed part of the current contract month.

- **Full model** (`full_model.py`): a step path that can only change on a scheduled FOMC
  meeting date. The size of each step is recovered one contract month at a time (a
  bootstrap) from the day-weighted-average SR1 pricing equation in Section 3 of
  `docs/METHODOLOGY.md` (PR #11). This is a scoped-down version of the fit in that
  document's Section 5, restricted to SR1 and to meeting dates as the only step points.
- **Simplified curve** (`simple_curve.py`): a smooth curve that knows nothing about FOMC
  dates. Ported from `build_short_rate_curve` in
  [`src/finm37000/futures.py`](https://github.com/pattersonem/finm37000-autumn-2026/blob/main/src/finm37000/futures.py),
  the course package's STIR-futures curve-building method (the "simplified SOFR
  curve-building approach similar to the one described in the course materials"
  mentioned in the team README). It bootstraps a continuously-compounded zero-rate curve
  from the SR1 strip, linearly interpolated between contract maturities, and reports the
  implied daily rate as that curve's local forward rate. It is reimplemented locally
  rather than imported, since the course package depends on `databento` at import time
  and this comparison makes no network calls at runtime.

Both operate in SOFR-rate space (since SR1 settles on SOFR, not the policy rate
directly); the known constant SOFR-policy spread (`world.spread_pct`, a synthetic-data
parameter) is subtracted before comparing either method's output to the true policy path.
A real analysis would have to estimate that spread instead (Section 2 of
`docs/METHODOLOGY.md`).

## Definitions

Matching `forecast_eval` (#5, PR #15) and `docs/METHODOLOGY.md` Section 9 (PR #11):

- **Next-meeting horizon**: for observation date `t`, the forecast target is the first
  scheduled FOMC meeting strictly after `t`. A meeting's own decision day is never its own
  target (that day's price is a nowcast, not a forecast).
- **Error** = implied policy rate just after the next meeting − true policy rate just
  after the next meeting, in bp. Positive means the method expected a higher rate than
  the (synthetic) Fed delivered.
- **Bias / MAE / RMSE**: mean error / mean absolute error / root mean squared error
  across observation dates, per method.
- **Days to meeting**: calendar days from the observation date to the next meeting,
  bucketed into weekly bins, to see whether a method's error shrinks as the meeting
  approaches.
- **Change vs. hold**: whether the next meeting's realized move was nonzero. The
  headline check for this issue is whether the simplified curve's error is concentrated
  around meetings with a rate change, versus holds.

## Synthetic data

`synthetic.py` generates, from one seed:

1. A true policy path: piecewise constant, 16 FOMC meetings over about two years (8 a
   year), with a hike-then-hold-then-cut pattern (+175 bp over 5 meetings, 6 holds, -150
   bp over 5 meetings).
2. Daily SOFR = true policy rate + a small constant spread + seeded daily noise.
3. SR1 settlement prices: 100 − the day-weighted average daily SOFR for each calendar
   month, with optional seeded price noise, using `sofr_pricing.sr1_rate` (#3, PR #12)
   on the synthetic daily SOFR series, which already has one value per calendar day.

Because the whole path is known in advance, SR1 prices are built with perfect foresight
of each contract month (the standard way to validate a curve-fitting method: generate
prices from a known path, then check that the method recovers it — see Section 12 of
`docs/METHODOLOGY.md`). `tests/test_curve_compare.py` checks that the full model recovers
the exact true deltas when there is no noise.

## Error before the first meeting

On the seeded demo run, the full model's error is concentrated on the 20 observation
dates before the first meeting (MAE 13.79 bp, RMSE 16.37 bp), compared with the
remaining 474 dates (MAE 0.63 bp, RMSE 0.96 bp). Across all 494 dates, its MAE is
1.17 bp and its RMSE is 3.43 bp. The overall accuracy table includes every date.

The first meeting (2024-01-31) falls at month end, leaving only one post-meeting day
in the 31-day January SR1 contract. For observations before that meeting, the bootstrap
solves its move from the January contract alone: it divides the difference between
the contract's total implied rate and the observed and baseline rate totals by that
one remaining meeting day. An error in the January contract's average rate is therefore
amplified 31 times in the inferred move; an anchor-rate error is amplified by the
number of unobserved days. Later contracts do not revise this move. A real-data run
should check how far the sample start precedes the first FOMC meeting it covers.

## Known limitation of the simplified curve

Plotting the simplified curve's implied path (`plot_sample_paths`) shows small jumps
at contract-month boundaries, not a perfectly smooth line. This is an inherent property of
bootstrapping a *linearly interpolated* zero-rate curve (the course method's own choice of
interpolation): the zero rate is continuous, but its slope changes at each node, so the
derived local forward rate can jump there.
It is not a bug in this port; the course package's own `build_short_rate_curve` has the
same property. A monotone-cubic interpolation would smooth it out but is not what the
course method does.

## Pending

- **Real-data integration**: connect collected SR1 settlements, daily SOFR, and fitted
  implied paths to the comparison. The demo currently supplies synthetic data in their
  place.
- **`fomc_data`**: replace the synthetic meeting schedule with dates from
  `fomc_data.fomc_calendar.fetch_fomc_calendar` for a real-data comparison.
- **`forecast_eval`**: this comparison implements only next-meeting, SR1-only horizon
  logic. A follow-up could feed both implied paths to `forecast_eval` for horizons 1-6.

## Usage

```python
from curve_compare import generate_synthetic_world, default_obs_dates, run_comparison, summarize_errors

world = generate_synthetic_world(seed=0)
results = run_comparison(world, default_obs_dates(world))
summarize_errors(results)   # bias / MAE / RMSE per method, at the next-meeting horizon
```

## How to run

From the repository root, with Python 3.12 and `pip install -r requirements.txt`:

```bash
python -m pytest tests/test_curve_compare.py -v        # unit tests
python -m scripts.run_curve_compare_demo                # tables on synthetic data
python -m scripts.run_curve_compare_demo --plot sample_paths.png   # also save both charts
```

No network access or data files are needed.
