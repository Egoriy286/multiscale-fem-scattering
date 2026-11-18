"""The validated reference training loop, with one joint train/val dataset.

No band routing or separate model/basis selection is added. A single bundle
contains one SVD-SNO v2, global k statistics and shared output POD spaces.
"""
from experiments.reference_sno_gmsfem.training import TrainConfig, run_experiment

__all__ = ["TrainConfig", "run_experiment"]
