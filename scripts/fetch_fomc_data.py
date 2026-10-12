"""
Run the FOMC and realized interest rate data pipeline.
"""

from fomc_data.config import ensure_dirs, EFFR_CSV, FOMC_CSV
from fomc_data.nyfed_io import fetch_effr
from fomc_data.dataset import build_fomc_decisions
from fomc_data.clean import validate_fomc_decisions


def main():
    """
    Fetch EFFR, build and validate FOMC decisions, and save both datasets.

    Returns: None. Writes EFFR and FOMC CSVs to the processed data directory.
    """
    ensure_dirs()

    #1. Download daily EFFR observations.
    print("Fetching EFFR data...")
    effr = fetch_effr()

    #2. Construct and validate FOMC decisions.
    print("Building FOMC decisions...")
    fomc = build_fomc_decisions()
    fomc = validate_fomc_decisions(fomc)

    #3. Save processed datasets. Keep both datasets separate for downstream modeling.
    effr.to_csv(EFFR_CSV, index = False)
    fomc.to_csv(FOMC_CSV, index = False)

    print(f"Saved {len(effr)} EFFR observations to {EFFR_CSV}")
    print(f"Saved {len(fomc)} FOMC meetings to {FOMC_CSV}")


if __name__ == "__main__":
    main()
