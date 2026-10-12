# FOMC Meeting and Realized Rate Data

## Overview

This pipeline collects historical Federal Open Market Committee (FOMC) meeting dates, realized federal funds target-rate decisions, and daily Effective Federal Funds Rate (EFFR) observations from January 2022 through September 2026.

The resulting datasets provide the realized monetary policy outcomes used to evaluate expectations implied by SOFR futures.

## Data Sources

| Dataset | Source | Description |
|---|---|---|
| FOMC meeting calendar | Federal Reserve Board | Historical scheduled FOMC policy announcement dates |
| Effective Federal Funds Rate (EFFR) | Federal Reserve Bank of New York | Daily observed overnight federal funds rate |
| Federal funds target range | FRED (`DFEDTARL`, `DFEDTARU`) | Daily lower and upper bounds of the Fed's target range, sourced from the Federal Reserve Board |

**Official sources:**
- https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- https://www.newyorkfed.org/markets/reference-rates/effr
- https://fred.stlouisfed.org/series/DFEDTARL
- https://fred.stlouisfed.org/series/DFEDTARU

## Data Processing

1. Retrieve historical EFFR observations from the New York Fed's public API.
2. Construct the historical FOMC announcement calendar for the sample period.
3. Download daily federal funds target-range bounds from FRED.
4. Match each FOMC announcement to the target range immediately before the meeting and the first observation afterward.
5. Calculate changes in the target midpoint in basis points and classify decisions as hikes, cuts, or holds.
6. Validate the resulting meeting-level dataset for duplicates, missing values, invalid target ranges, and inconsistent policy changes.

The current meeting calendar covers regularly scheduled FOMC meetings. Unscheduled policy actions are not separately represented.

The effective dates inferred from the target-rate series are provisional until verified against official Federal Reserve implementation notes.

## Reproducing the Data

From the repository root, run:

```bash
python -m scripts.fetch_fomc_data
```

This generates:

- `data/processed/effr_daily.csv` — daily EFFR observations.
- `data/processed/fomc_meetings.csv` — FOMC meetings and realized target-rate decisions.

Generated datasets are excluded from Git through `.gitignore` and can be recreated by running the pipeline.

## Alignment with SOFR Futures

The pipeline includes an alignment module (`fomc_data/alignment.py`) that maps SOFR futures settlement observations to upcoming FOMC policy decisions based on when the decisions take effect within each contract's reference period.

The alignment uses the cleaned SOFR futures dataset produced by Issue #1:

`data/processed/sofr_futures_settlements.parquet`

For each futures observation, the function identifies FOMC meetings satisfying:

- The FOMC announcement date is strictly after the futures observation date.
- The policy `effective_date` falls within the contract's reference period, including its boundaries.

The `meeting_date` is used for the pre-announcement filter (`trade_date < meeting_date`), while `effective_date` is used for contract overlap (`reference_start <= effective_date <= reference_end`). This distinction matters when an announcement occurs at the end of a month but the policy change takes effect in the next month.

For hold decisions with no recorded `effective_date`, the alignment uses the next business day after the meeting as a fallback. This fallback uses pandas business-day conventions, which exclude weekends but do not explicitly account for Federal Reserve holidays. For simplified input data without an `effective_date` column, the same fallback is applied; production rate-change dates should be populated and verified against official implementation notes.

The resulting dataset preserves the original futures information and attaches the realized policy decision, target-rate bounds, target midpoint, and rate change in basis points.

Multiple FOMC meetings may be associated with a single futures contract observation. Contracts without a matching meeting are omitted from the alignment output but remain available in the original futures dataset.

### Running the Alignment

First, generate the FOMC datasets:

```bash
python -m scripts.fetch_fomc_data
```

Once the SOFR futures data pipeline has generated its cleaned settlement file, run:

```bash
python -m scripts.align_fomc_futures
```

This creates:

`data/processed/fomc_futures_alignment.csv`

The alignment dataset provides a meeting-to-contract mapping for subsequent pricing and forecast-evaluation stages. It does not itself estimate market-implied policy rates or calculate forecast errors.

**Important:** The alignment has been tested using synthetic SOFR futures observations. End-to-end integration with the actual Databento settlement dataset remains to be verified once that file is available locally.

## Validation

Run the automated tests:

```bash
python -m pytest tests/test_fomc_data.py -v
```

The seven automated tests check meeting-calendar ordering and uniqueness, acceptance of a valid policy decision, rejection of an incorrectly calculated basis-point change, and synthetic SOFR futures alignment. Alignment tests include multiple meetings within one contract, a month-end announcement whose effective date falls in the next contract period, and the fallback for a hold without a recorded effective date.

These tests establish internal consistency but do not independently verify the historical policy decisions against FOMC statements.

## Integration with SOFR Futures

These datasets complement the SOFR futures settlement data collected separately through Databento.

The FOMC announcement dates define the policy events and the pre-announcement cutoff; policy effective dates determine contract reference-period overlap. Realized target rates provide the outcomes against which market-implied expectations can be evaluated.

Forecast comparisons must use information available before each FOMC announcement to avoid look-ahead bias.