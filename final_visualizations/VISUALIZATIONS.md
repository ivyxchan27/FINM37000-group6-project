# Final Visualizations and Analysis

## Overview

This module produces the figures and summary analyses for Issue #7 of the *Pricing the Fed with SOFR Futures* project. It compares market-implied Federal Reserve policy paths with realized FOMC decisions, evaluates forecast errors across meeting horizons, and compares the full SOFR futures methodology with a simplified curve approach.

The figures are intended for the final report and presentation. The visualization pipeline consumes the outputs of the preceding data, estimation, and evaluation modules; it does not independently retrieve futures prices or estimate the market-implied policy path.

## Data Inputs

| Dataset | Source | Description |
|---|---|---|
| Realized FOMC decisions | Issue #2 (`fomc_data`) | Meeting dates, policy effective dates, and target-rate bounds |
| Market-implied policy paths | Issue #4 (`sofr_pricing`) | Forecast observation dates and implied meeting-level or cumulative policy changes |
| Forecast evaluation | Issue #5 (`forecast_eval`) | Realized changes, signed forecast errors, and forecast horizons |
| Full versus simplified comparison | Issue #6 (`curve_compare`) | Matched forecast errors from both methodologies and FOMC decision classifications |

Forecast observations are associated with an observation date (`obs_date`) and a forecasted FOMC meeting (`meeting_date`). Forecast horizons count the number of FOMC meetings ahead. Policy-rate levels are expressed in percent, while forecast errors and policy changes are expressed in basis points (bp).

## Figure Construction

The pipeline produces four principal visualizations:

1. **Market-implied versus realized policy paths:** A step chart compares the policy-rate path implied by futures on a selected observation date with subsequent realized target-rate midpoints. The shaded area illustrates the difference between the paths; it is not a confidence interval.
2. **Forecast errors by FOMC horizon:** Boxplots show the distribution of signed errors, while an adjacent panel reports mean absolute error (MAE), root mean squared error (RMSE), and bias across available forecast horizons.
3. **Full versus simplified methodology:** Grouped bars compare overall MAE and RMSE. A dumbbell plot compares the methods' MAE separately for FOMC rate changes and holds.
4. **Notable FOMC forecasting episodes:** Meetings are selected using the largest and smallest meeting-level MAE and the largest positive and negative mean bias, where those categories exist. An additional event-study plotting function can display how forecasts evolved ahead of a selected meeting when sufficient observation-level data are available.

The notable-meetings categories are not required to identify distinct meetings. If only one meeting qualifies, it may appear in multiple categories. A negative-bias category is omitted when no qualifying meeting has negative mean forecast error.

## Forecast Accuracy Measures

For each forecast, the signed error is defined as the market-implied policy change minus the corresponding realized policy change, in basis points.

- **MAE:** The mean absolute forecast error. Lower values indicate smaller average errors.
- **RMSE:** The square root of the mean squared forecast error. This places greater weight on large misses.
- **Bias:** The mean signed forecast error. Positive values indicate that implied changes exceed realized changes on average; negative values indicate the reverse.

Statistics are calculated from the observations in each reported group. Comparisons between the full and simplified models should use the same underlying observation sample to ensure that differences reflect methodology rather than sample composition.

## Reproducing the Figures

From the repository root, run the visualization tests:

```bash
python -m pytest tests/test_fed_paths.py tests/test_model_comparison.py tests/test_summary_tables.py tests/test_historical_examples.py -v
```

Run the figure-generation test to exercise the end-to-end synthetic example:

```bash
python -m pytest tests/test_generate_final_figures.py -v -s
```

The figure-generation entry point is `scripts/generate_final_figures.py`. It uses the plotting functions in `final_visualizations/`, with shared Matplotlib formatting defined in `final_visualizations/style.py`. Saved figure locations and any exported tables should be checked against the script's configured output directory.

## Validation

The plotting functions have been exercised using synthetic forecast and comparison data. The tests cover figure creation, rate-level construction, model-comparison summaries, notable-meeting selection, and figure generation.

These tests establish that the visualization code can process the example inputs. They do not independently verify the historical SOFR futures estimates, realized FOMC outcomes, or the accuracy of the final empirical conclusions.

**Important:** The current example figures are synthetic illustrations, not historical estimates. In particular, the single-horizon forecast-error chart and repeated notable-meeting selections reflect limited synthetic coverage and should not be interpreted as empirical findings.

## Integration with Forecast Evaluation

The final figures depend on the upstream SOFR futures estimation, forecast-evaluation, and model-comparison pipelines. Before using them in the final report, the integrated workflow must be checked for consistent column definitions, matching observation samples, and the distinction between FOMC announcement dates and policy effective dates.

Forecast comparisons must use only information available before each relevant FOMC announcement to avoid look-ahead bias. The historical version of the horizon chart should cover the available 1–6 meeting-ahead forecasts, and the notable-episodes analysis should be regenerated using actual observation-level forecasts.

End-to-end integration with the real upstream outputs remains to be verified. Until that validation is complete, synthetic figures should be labeled as illustrative and should not be presented as evidence of historical forecasting performance.
