"""Same static data as the full-SVD profile, with separate randomized runs."""

from pathlib import Path

from ..settings import (N_FREQUENCIES, N_DIRECTIONS, DATASET_TAG,
                        LOW_DATA_DIR, HIGH_DATA_DIR)

BASE = Path(__file__).resolve().parent
POD_METHOD = "randomized"
LOW_RUNS = BASE / "runs" / f"low_frequency_{DATASET_TAG}"
HIGH_RUNS = BASE / "runs" / f"high_frequency_{DATASET_TAG}"
SVD_RUNS = BASE / "runs" / f"svd_diagnostics_{DATASET_TAG}"
COMPARISON_RUNS = BASE / "runs" / f"comparison_{DATASET_TAG}"
