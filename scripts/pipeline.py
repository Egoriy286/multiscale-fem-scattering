"""Run reproducible experiments through the complex FEM Docker service."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ["bash", str(ROOT / "scripts/docker.sh")]
SERVICE = "dolfinx-complex"
STAGES = ("prepare", "train", "baselines", "verify", "report")
EXPERIMENTS = ("mscale_fno_100", "mscale_sno_100", "shared_sno_gmsfem")
SEEDED_EXPERIMENTS = ("mscale_fno_100", "mscale_sno_100")
RUNTIME = """import json, platform
import dolfinx, gmsh, numpy as np, scipy, torch
from petsc4py import PETSc
assert np.dtype(PETSc.ScalarType).kind == 'c', 'Complex PETSc required'
torch.fft.rfft2(torch.ones(4, 4))
print(json.dumps(dict(python=platform.python_version(), machine=platform.machine(),
    dolfinx=dolfinx.__version__, petsc=list(PETSc.Sys.getVersion()),
    scalar=str(np.dtype(PETSc.ScalarType)), torch=str(torch.__version__),
    numpy=np.__version__, scipy=scipy.__version__, gmsh=gmsh.__version__)))
"""


def docker_exec(*args):
    return DOCKER + ["compose", "exec", "-T", "-e", "OMP_NUM_THREADS=2",
                     "-e", "OPENBLAS_NUM_THREADS=2", SERVICE, *args]


def command_for(stage, experiment, epochs=None, seeds=None):
    command = docker_exec("python3", "-u", "-m", f"experiments.{experiment}.run", stage)
    if stage == "train":
        if epochs is not None:
            command += ["--epochs", str(epochs)]
        if seeds is not None:
            if experiment not in SEEDED_EXPERIMENTS:
                raise ValueError("--seeds requires a multiscale FNO or SNO experiment")
            command += ["--seeds", *map(str, seeds)]
    return command


def test_command(experiment):
    suites = ["tests", f"experiments/{experiment}/tests"]
    if experiment != "mscale_sno_100":
        suites.append("experiments/mscale_sno_100/tests")
    suites.append("experiments/multiscale_sample_study/tests")
    return docker_exec("python3", "-m", "pytest", "--import-mode=importlib", *suites,
        "experiments/reference_sno_gmsfem/tests", "experiments/gms_sno_100/tests",
        "neural_operator/tests/test_operators.py", "neural_operator/tests/test_pod_fem_loss.py",
        "gmsfem/tests", "-q")


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def run_logged(command, log):
    """Keep complete output on disk and stream it for live/CI visibility."""
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            return process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise


def execute(args):
    directory = ROOT / "experiments" / args.experiment
    log_root = directory / "pipeline"
    log_root.mkdir(parents=True, exist_ok=True)
    # flock is released by the OS after termination, including SIGKILL.
    with (log_root / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another pipeline is running for {args.experiment}") from None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        logs = log_root / stamp
        logs.mkdir()
        manifest = dict(experiment=args.experiment, started_at=stamp, status="running", stages=[])
        latest = log_root / "latest.json"
        manifest["log_directory"] = str(logs.relative_to(ROOT))
        atomic_json(latest, manifest)
        tasks = []
        if args.build:
            tasks.append(("build", DOCKER + ["compose", "build", SERVICE]))
        tasks += [("up", DOCKER + ["compose", "up", "-d", "--wait", "--wait-timeout", "180", SERVICE]),
                  ("environment", docker_exec("python3", "-c", RUNTIME))]
        if not args.skip_tests:
            tasks.append(("tests", test_command(args.experiment)))
        tasks += [(stage, command_for(stage, args.experiment, args.epochs, args.seeds))
                  for stage in STAGES[STAGES.index(args.from_stage):]]
        try:
            for name, command in tasks:
                row = dict(stage=name, status="running", command=command,
                           log=str((logs / f"{name}.log").relative_to(ROOT)))
                manifest["stages"].append(row)
                atomic_json(latest, manifest)
                started = time.monotonic()
                print(f"\n>>> {name}", flush=True)
                code = run_logged(command, logs / f"{name}.log")
                row.update(returncode=code, seconds=time.monotonic()-started,
                           status="passed" if code == 0 else "failed")
                atomic_json(latest, manifest)
                if code:
                    raise RuntimeError(f"Stage {name} failed (exit {code}); see {row['log']}")
            manifest["status"] = "passed"
        except BaseException:
            manifest["status"] = "failed"
            if manifest["stages"] and manifest["stages"][-1]["status"] == "running":
                manifest["stages"][-1]["status"] = "interrupted"
            raise
        finally:
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(latest, manifest)
            atomic_json(logs / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "doctor", "test", "status"))
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="mscale_fno_100")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--from-stage", choices=STAGES, default="prepare")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.seeds is not None and (args.experiment not in SEEDED_EXPERIMENTS or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0):
        parser.error("--seeds requires a multiscale FNO/SNO experiment and distinct nonnegative integers")
    if args.action == "status":
        path = ROOT / "experiments" / args.experiment / "pipeline/latest.json"
        print(path.read_text() if path.exists() else "Pipeline has not been run")
    elif args.action == "doctor":
        subprocess.run(DOCKER + ["compose", "ps"], cwd=ROOT, check=True)
        subprocess.run(docker_exec("python3", "-c", RUNTIME), cwd=ROOT, check=True)
    elif args.action == "test":
        subprocess.run(test_command(args.experiment), cwd=ROOT, check=True)
    else:
        execute(args)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
