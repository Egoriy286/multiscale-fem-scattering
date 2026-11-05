"""Shared dataset sizes and local artifact paths for the current experiment."""

from pathlib import Path

BASE = Path(__file__).resolve().parent
N_MEDIA = 1000
N_FREQUENCIES = 1000
N_DIRECTIONS = 8
DATASET_TAG = f"static_{N_FREQUENCIES}x{N_DIRECTIONS}"
POD_METHOD = "full"
LOW_DATA_DIR = BASE / "data" / f"static_low_frequency_{N_FREQUENCIES}x{N_DIRECTIONS}"
HIGH_DATA_DIR = BASE / "data" / f"static_high_frequency_{N_FREQUENCIES}x{N_DIRECTIONS}"
LOW_RUNS = BASE / "runs" / LOW_DATA_DIR.name
HIGH_RUNS = BASE / "runs" / HIGH_DATA_DIR.name
SVD_RUNS = BASE / "runs" / f"svd_diagnostics_{DATASET_TAG}"
COMPARISON_RUNS = BASE / "runs" / f"comparison_{DATASET_TAG}"
