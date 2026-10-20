"""
Align FOMC policy decisions with SOFR futures settlement data.
"""

import pandas as pd
from fomc_data.alignment import align_fomc_with_futures
from fomc_data.config import (FOMC_CSV, SOFR_FUTURES_PARQUET, FOMC_ALIGNMENT_CSV, ensure_dirs)


def main():
    """
    Load futures and FOMC data, align meetings, and save the mapping.

    Returns: None. Writes the alignment dataset to the processed data directory.
    """
    ensure_dirs()

    if not FOMC_CSV.exists():
        raise FileNotFoundError("FOMC dataset not found. Run python -m scripts.fetch_rate_data first.")

    if not SOFR_FUTURES_PARQUET.exists():
        raise FileNotFoundError("SOFR futures dataset not found. Run the SOFR futures data pipeline first.")

    fomc = pd.read_csv(FOMC_CSV, parse_dates = ["meeting_date"])
    futures = pd.read_parquet(SOFR_FUTURES_PARQUET)

    aligned = align_fomc_with_futures(futures, fomc)
    aligned.to_csv(FOMC_ALIGNMENT_CSV, index = False)

    print(f"Saved {len(aligned)} aligned observations to {FOMC_ALIGNMENT_CSV}")


if __name__ == "__main__":
    main()
