"""Render the current static SVD-SNO prediction figures.

From the repository root:
    docker compose exec -T dolfinx-complex python paper/svd_sno_revision_2026-11-06/prepare_predictions.py

The archived variable-medium pilot is available only through the explicit
prepare_pilot_predictions.py script. No training or FEM solves are performed.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("prepare_static_predictions.py")), run_name="__main__")
