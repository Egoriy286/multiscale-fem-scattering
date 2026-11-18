"""Train one shared SVD-SNO and compare it with paired reference GMsFEM."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from experiments.reference_sno_gmsfem.dataset import _sha256, load_dataset
from experiments.reference_sno_gmsfem.run import (
    evaluation_rows, training_config, write_csv, write_json,
)
from .dataset import prepare_shared_dataset

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "reference_sno_gmsfem"
DATA = HERE / "data"
RUN = HERE / "runs" / "shared"
RESULTS = HERE / "results"


def environment():
    import platform
    from datetime import datetime, timedelta, timezone
    import dolfinx
    import gmsh
    import numpy as np
    from petsc4py import PETSc
    import scipy
    import torch

    value = dict(
        recorded_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        timezone="Asia/Yakutsk", python=platform.python_version(), machine=platform.machine(),
        numpy=np.__version__, scipy=scipy.__version__, torch=str(torch.__version__),
        dolfinx=dolfinx.__version__, gmsh=gmsh.__version__,
        petsc=list(PETSc.Sys.getVersion()), scalar_type=str(np.dtype(PETSc.ScalarType)),
        device="cpu", training_threads=2,
        source_sha256={str(p.relative_to(HERE)): _sha256(p)
                       for p in sorted(HERE.rglob("*.py"))},
    )
    if value["scalar_type"] != "complex128":
        raise RuntimeError("Complex PETSc is required")
    write_json(RESULTS / "environment.json", value)


def prepare():
    data = prepare_shared_dataset(DATA, SOURCE / "data")
    for name in ("config", "dataset_info", "provenance"):
        write_json(RESULTS / f"dataset_{name}.json",
                   json.loads((DATA / f"{name}.json").read_text()))
    return data


def train(args):
    from .training import run_experiment
    data = load_dataset(DATA)
    config = training_config(args)
    protocol = {
        "model_count": 1, "k_range": [.1, 4.], "config": asdict(config),
        "selection": "POD rank and checkpoint chosen only on inherited validation",
        "source_splits_preserved": True,
        "dataset_config_sha256": _sha256(DATA / "config.json"),
        "dataset_info_sha256": _sha256(DATA / "dataset_info.json"),
        "source_sha256": {str(p.relative_to(HERE.parents[1])): _sha256(p) for p in (
            HERE / "dataset.py", HERE / "training.py", SOURCE / "training.py",
            HERE.parents[1] / "neural_operator/training.py",
            HERE.parents[1] / "neural_operator/sno/model.py",
            HERE.parents[1] / "neural_operator/sno/basis.py",
            HERE.parents[1] / "neural_operator/_layers.py",
            HERE.parents[1] / "neural_operator/losses.py",
        )},
    }
    path = RESULTS / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Frozen shared training protocol differs; use a new experiment")
    write_json(path, protocol)
    predictor, metrics, history = run_experiment(data, RUN, config)
    write_json(RESULTS / "sno_metrics.json", metrics)
    write_json(RESULTS / "sno_history.json", history)
    rows = evaluation_rows(data, predictor, data.split_indices()[2], "SVD-SNO shared")
    write_json(RESULTS / "sno.json", {"rows": rows})
    write_csv(RESULTS / "sno.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "baselines", "verify", "report", "visualize", "environment", "all"))
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()
    if args.stage in ("environment", "all"):
        environment()
    if args.stage in ("prepare", "all"):
        prepare()
    if args.stage in ("train", "all"):
        train(args)
    if args.stage in ("baselines", "all"):
        from .comparison import import_baselines
        import_baselines(DATA, SOURCE, RESULTS)
    if args.stage in ("verify", "all"):
        from .comparison import verify_predictions
        verify_predictions(DATA, RUN, RESULTS)
    if args.stage in ("report", "all"):
        from .report import build_report
        build_report(DATA, RUN, RESULTS)
    if args.stage == "visualize":
        from .visualize import build_visuals
        build_visuals(DATA, RUN, RESULTS)


if __name__ == "__main__":
    main()
