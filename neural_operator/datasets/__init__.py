"""Plan, generate, inspect, and load the shared scattering datasets.

The default experiment fixes medium 0 (seed 2026) and varies frequency and
direction. DatasetConfig retains the older variable-material experiment.
Importing this package never loads Gmsh, DOLFINx, or the FEM solver module.
"""
from ..data import inspect_dataset, load_dataset, write_dataset_info
from .config import DatasetConfig, StaticDatasetConfig, load_config, plan_dataset
from .generation import generate_dataset, generate_static_dataset

__all__ = ['DatasetConfig', 'StaticDatasetConfig', 'load_config', 'plan_dataset',
           'generate_dataset', 'generate_static_dataset', 'load_dataset',
           'write_dataset_info', 'inspect_dataset']
