"""Fresh native-mesh SVD-SNO training for the 2026-11-16 stability study.

Each mesh gets an independent train-only mass POD and a fresh, identically
configured network. The saved predictions remain in each mesh's native CG2
DOF ordering; cross-mesh field comparison is a separate common-quadrature step.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import torch

from experiments.gms_sno_100.model import SpectralSurrogate
from experiments.gms_sno_100.pod import ComplexPOD
from experiments.highk_svd_gmsfem.training import fit_streaming_pod
from experiments.multiscale_sample_study.sno import CoefficientLoss, model_config, _norm2
from experiments.multiscale_sample_study.training import _model_state, _save, parameter_count
from experiments.reference_sno_gmsfem.dataset import _write_json

ROOT = Path(__file__).resolve().parent
LEVELS = (0, 1, 2)
DEFAULT_PROTOCOL = dict(
    schema_version=1,
    train_count=1000,
    validation_count=200,
    test_count=200,
    seeds=[11, 29, 47],
    output_modes=192,
    width=8,
    depth=4,
    scales=[1, 40, 80, 100, 120, 140, 180, 200],
    activation="sine",
    epochs=1000,
    batch_size=100,
    learning_rate=0.001,
    weight_decay=0.0001,
    validation_interval=5,
    patience_checks=40,
    device="mps",
    threads=2,
    pod_batch_size=16,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_manifest():
    repo = Path(__file__).resolve().parents[2]
    files = [Path(__file__).resolve(),
        repo/"experiments/highk_svd_gmsfem/training.py",
        repo/"experiments/gms_sno_100/model.py",
        repo/"experiments/gms_sno_100/pod.py",
        repo/"experiments/multiscale_sample_study/sno.py",
        repo/"experiments/multiscale_sample_study/training.py"]
    return {str(path.relative_to(repo)):sha256(path) for path in files}


def _validate_protocol(protocol):
    supplied = dict(protocol)
    aliases = {"pod_rank": "output_modes", "lr": "learning_rate",
               "val_interval": "validation_interval", "n_train": "train_count",
               "n_validation": "validation_count", "n_test": "test_count"}
    for source, target in aliases.items():
        if source in supplied and target not in supplied:
            supplied[target] = supplied[source]
    sizes = supplied.get("train_sizes", [supplied.get("train_count", 1000)])
    if sizes not in ([1000], [2000]):
        raise ValueError("run one declared training size per phase: [1000] or [2000]")
    supplied["train_sizes"] = sizes
    supplied["train_count"] = sizes[0]
    required = dict(DEFAULT_PROTOCOL)
    required.update(supplied)
    if required.get("levels", [0, 1, 2]) != [0, 1, 2]:
        raise ValueError("this primary phase requires levels [0,1,2]")
    required["levels"] = [0, 1, 2]
    for key in ("train_count", "validation_count", "test_count", "output_modes",
                "width", "depth", "epochs", "batch_size", "validation_interval",
                "patience_checks", "threads", "pod_batch_size"):
        value = required[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if required["output_modes"] != 192:
        raise ValueError("This frozen study protocol requires exactly 192 complex output modes")
    if required["width"] != 8 or required["depth"] != 4:
        raise ValueError("This frozen study protocol requires width 8 and depth 4")
    if required["scales"] != DEFAULT_PROTOCOL["scales"] or required["activation"] != "sine":
        raise ValueError("This frozen study protocol requires the declared multiscale sine architecture")
    if required["seeds"] != [11, 29, 47]:
        raise ValueError("This frozen study protocol requires seeds 11, 29, and 47")
    if required["device"] not in ("mps", "cpu"):
        raise ValueError("device must be mps or cpu")
    if required["device"] == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    for key in ("learning_rate", "weight_decay"):
        if not math.isfinite(float(required[key])) or float(required[key]) < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if required["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    return required


class NativeDataset:
    """Memory-mapped native CG2 labels and mesh-local mass matrix."""
    def __init__(self, root, level, queries):
        self.root = Path(root)
        self.level = int(level)
        directory = self.root / "data" / f"level{self.level}"
        with np.load(directory / "geometry.npz", allow_pickle=False) as geometry:
            self.coordinates = geometry["coordinates"]
            self.triangles = geometry["triangles"]
            self.cell_tags = geometry["cell_tags"]
        self.mass = sp.load_npz(directory / "mass.npz").tocsr()
        self.fields = np.load(directory / "fields.npy", mmap_mode="r", allow_pickle=False)
        self.queries = queries
        self.k0 = np.asarray([float(q.get("k", q.get("k0"))) for q in queries])
        self.theta = np.asarray([float(q.get("theta", q.get("angle"))) for q in queries])
        self.split = np.asarray([int(q["split"]) for q in queries])
        nodes = len(self.coordinates)
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
            raise ValueError("geometry coordinates must have shape [CG2 DOFs,2]")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 6:
            raise ValueError("native CG2 connectivity must have six DOFs per triangle")
        if self.mass.shape != (nodes, nodes):
            raise ValueError("native CG2 mass matrix does not match geometry")
        if self.fields.shape != (len(queries), nodes) or self.fields.dtype != np.complex128:
            raise ValueError("fields.npy must be complex128 [query count,native CG2 DOFs]")
        if self.split.shape != (len(queries),) or not np.isin(self.split, (0, 1, 2)).all():
            raise ValueError("query split must contain 0=train, 1=validation, 2=test")
        if not np.isfinite(self.k0).all() or not np.isfinite(self.theta).all():
            raise ValueError("query coordinates must be finite")

    def split_indices(self):
        return tuple(np.flatnonzero(self.split == i) for i in range(3))


def load_queries(root, filename="queries.json"):
    path = Path(root) / filename
    raw = json.loads(path.read_text())
    rows = raw["queries"] if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or not rows:
        raise ValueError("queries.json must contain a nonempty query list")
    for i, row in enumerate(rows):
        if (not isinstance(row, dict) or "split" not in row
                or not any(key in row for key in ("k", "k0"))
                or not any(key in row for key in ("theta", "angle"))):
            raise ValueError(f"query row {i} must contain k/k0, theta/angle, and split")
    return rows, sha256(path)


def _encode(pod, fields, mass):
    fields = np.asarray(fields, dtype=np.complex128)
    centered = fields - pod.mean
    weighted = mass @ centered.T
    return ((weighted.T.conj() @ pod.modes).conj())


def _cache_directory(root, level, train_count):
    suffix = f"level{level}" if int(train_count) == 1000 else f"n{int(train_count)}/level{level}"
    return Path(root)/"cache"/suffix


def _projection_cache(root, data, protocol, queries_hash):
    train_count = protocol["train_count"]
    directory = _cache_directory(root, data.level, train_count)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "preprocessing.json"
    specification = dict(protocol_sha256=json_hash(protocol), level=data.level,
                         mesh_sha256=sha256(Path(root)/"meshes"/f"level{data.level}.msh"),
                         queries_sha256=queries_hash,
                         geometry_sha256=sha256(Path(root)/"data"/f"level{data.level}"/"geometry.npz"),
                         mass_sha256=sha256(Path(root)/"data"/f"level{data.level}"/"mass.npz"),
                         fields_sha256=sha256(Path(root)/"data"/f"level{data.level}"/"fields.npy"),
                         pod_source_sha256=sha256(Path(__file__).resolve().parents[1]/"highk_svd_gmsfem"/"training.py"),
                         train_ids=data.split_indices()[0][:train_count].tolist(),
                         train_count=train_count, rank=protocol["output_modes"])
    files = [directory/name for name in ("pod.npz", "coefficients.npy", "floor_relative_l2.npy",
                                          "norm2.npy", "preprocessing.npz")]
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved.get("specification") != specification:
            raise ValueError(f"level {data.level} preprocessing belongs to a different data/protocol")
        for path in files:
            expected = saved["files"].get(path.name)
            if not expected or sha256(path) != expected:
                raise ValueError(f"level {data.level} preprocessing artifact changed: {path.name}")
        return saved

    train = data.split_indices()[0][:train_count]
    pod, diagnostics = fit_streaming_pod(data.fields, data.mass, train,
        max_rank=protocol["output_modes"], relative_tolerance=.999999,
        batch_size=protocol["pod_batch_size"], expansion_rank=protocol["output_modes"],
        candidate_ranks=[protocol["output_modes"]])
    if pod.rank != protocol["output_modes"]:
        raise ValueError(f"level {data.level} train-only numerical POD rank {pod.rank}; expected exactly {protocol['output_modes']}")
    coefficients = np.lib.format.open_memmap(directory/"coefficients.npy", mode="w+",
        dtype=np.complex128, shape=(len(data.queries), pod.rank))
    floor = np.lib.format.open_memmap(directory/"floor_relative_l2.npy", mode="w+",
        dtype=np.float64, shape=(len(data.queries),))
    norm2 = np.lib.format.open_memmap(directory/"norm2.npy", mode="w+",
        dtype=np.float64, shape=(len(data.queries),))
    for start in range(0, len(data.queries), protocol["pod_batch_size"]):
        ids = np.arange(start, min(start+protocol["pod_batch_size"], len(data.queries)))
        fields = np.asarray(data.fields[ids])
        c = _encode(pod, fields, data.mass)
        residual = fields - pod.decode(c)
        coefficients[ids] = c
        norm2[ids] = _norm2(fields, data.mass)
        floor[ids] = np.sqrt(_norm2(residual, data.mass) / np.maximum(norm2[ids], 1e-300))
    coefficients.flush(); floor.flush(); norm2.flush()
    scale = float(np.sqrt(np.mean(np.abs(coefficients[train])**2)))
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(floor).all():
        raise ValueError("nonfinite or zero train-only coefficient scaling/projection floor")
    kmean = float(data.k0[train].mean()); kstd = float(data.k0[train].std())
    if kstd <= 0: raise ValueError("training query frequencies must vary")
    inputs = np.column_stack(((data.k0-kmean)/kstd, np.cos(data.theta), np.sin(data.theta))).astype(np.float32)
    pairs = np.stack((coefficients.real, coefficients.imag), axis=-1).astype(np.float32)
    np.savez(directory/"preprocessing.npz", inputs=inputs, pairs=pairs, scale=scale,
             k_mean=kmean, k_std=kstd)
    pod.save(directory/"pod.npz")
    del coefficients, floor, norm2
    diagnostics.update(projection_floor_pct={name:dict(mean=float(np.mean(np.load(directory/'floor_relative_l2.npy',mmap_mode='r')[ids]))*100,
        rms=float(np.sqrt(np.mean(np.asarray(np.load(directory/'floor_relative_l2.npy',mmap_mode='r')[ids])**2)))*100,
        p95=float(np.percentile(np.load(directory/'floor_relative_l2.npy',mmap_mode='r')[ids],95))*100)
        for name,ids in zip(("train","validation","test"),data.split_indices())})
    result=dict(specification=specification, diagnostics=diagnostics,
                files={path.name:sha256(path) for path in files})
    _write_json(marker, result)
    return result


class Adapter:
    def __init__(self, root, level, seed, protocol):
        self.directory = _cache_directory(root,level,protocol["train_count"])
        self.device = torch.device(protocol["device"])
        with np.load(self.directory/"preprocessing.npz", allow_pickle=False) as prep:
            self.inputs = torch.as_tensor(prep["inputs"], device=self.device)
            self.scale = float(prep["scale"])
            self.pairs = prep["pairs"]
        self.pod = ComplexPOD.load(self.directory/"pod.npz")
        self.config = model_config(self.pod.rank, "mscale_sno", protocol["width"], protocol["depth"])
        if self.config["scales"] != protocol["scales"] or self.config["activation"] != protocol["activation"]:
            raise ValueError("model factory differs from frozen architecture")
        torch.manual_seed(int(seed))
        self.model = SpectralSurrogate(**self.config).to(self.device)
        prep_obj = dict(pairs=self.pairs, scale=self.scale,
                        residual2=np.asarray(np.load(self.directory/"floor_relative_l2.npy",mmap_mode="r"))**2 *
                            np.asarray(np.load(self.directory/"norm2.npy",mmap_mode="r")))
        prep_obj["norm2"] = np.load(self.directory/"norm2.npy",mmap_mode="r")
        self.loss = CoefficientLoss(prep_obj, protocol["device"], torch.float32)
        self.validation_loss = CoefficientLoss(prep_obj, "cpu", torch.float64)

    @torch.no_grad()
    def values(self, ids, batch_size):
        self.model.eval()
        return np.concatenate([self.validation_loss(self.model(self.inputs[batch]), batch).cpu().numpy()
                               for batch in _batches(ids, batch_size)])

    @torch.no_grad()
    def predict(self, ids, batch_size=16):
        self.model.eval()
        for batch in _batches(ids, batch_size):
            pairs = self.model(self.inputs[batch]).detach().cpu().numpy().astype(np.float64)*self.scale
            coeff = pairs[..., 0] + 1j*pairs[..., 1]
            yield batch, self.pod.decode(coeff)


def _batches(ids, size):
    ids = np.asarray(ids, dtype=np.int64)
    return (ids[start:start+size] for start in range(0, len(ids), size))


def run_directory(root, level, seed, train_count=1000):
    if int(train_count) == 1000:
        return Path(root)/"runs"/f"level{level}"/f"seed_{seed}"
    return Path(root)/"runs"/f"n{int(train_count)}"/f"level{level}"/f"seed_{seed}"


def _train_one(root, data, protocol, seed, protocol_hash, queries_hash, preprocessing_hash, sources):
    all_train, validation, test = data.split_indices()
    train=all_train[:protocol["train_count"]]
    if len(train)!=protocol["train_count"]: raise ValueError("not enough train queries for declared count")
    directory = run_directory(root, data.level, seed,len(train)); directory.mkdir(parents=True, exist_ok=True)
    specification = dict(protocol_sha256=protocol_hash, queries_sha256=queries_hash,
                         preprocessing_sha256=preprocessing_hash,
                         source_sha256=sources,
                         level=data.level, seed=seed, train_count=len(train),training_ids=train.tolist(),
                         validation_ids=validation.tolist(), test_ids=test.tolist())
    result_path = directory/"result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["specification"] != specification: raise ValueError("existing run protocol differs")
        if any(sha256(directory/name) != result[key] for name,key in
               (("model.pt","checkpoint_sha256"),("predictions.npy","predictions_sha256"),
                ("metrics.json","metrics_sha256"),("history.json","history_sha256"),
                ("config.json","config_sha256"))):
            raise ValueError("existing run artifact hash mismatch")
        return result
    adapter = Adapter(root, data.level, seed, protocol)
    parameters = parameter_count(adapter.model)
    config = dict(specification=specification, source_sha256=sources, model=adapter.config,
                  parameters=parameters, output_modes=adapter.pod.rank,
                  support_dofs=len(data.coordinates), native_element="CG2",
                  initialization="fresh random initialization; no inherited checkpoint")
    config_path = directory/"config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("existing checkpoint configuration differs")
    _write_json(config_path, config)
    optimizer = torch.optim.Adam(adapter.model.parameters(), lr=protocol["learning_rate"],
                                 weight_decay=protocol["weight_decay"], foreach=False)
    batches = int(np.ceil(len(train)/protocol["batch_size"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=protocol["epochs"]*batches)
    rng = np.random.default_rng(seed); first=1; history=[]; best=float("inf"); best_epoch=0; bad=0; steps=0; elapsed=0.; best_state=None
    resume = directory/"resume.pt"
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["specification"] != specification: raise ValueError("resume protocol differs")
        adapter.model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        rng.bit_generator.state=state["numpy_rng"]; torch.set_rng_state(state["torch_rng"])
        first=state["epoch"]+1; history=state["history"]; best=state["best"]; best_epoch=state["best_epoch"]; bad=state["bad"]; steps=state["steps"]; elapsed=state["elapsed"]; best_state=state["best_state"]
    started=perf_counter()
    for epoch in range(first, protocol["epochs"]+1):
        if bad >= protocol["patience_checks"]: break
        adapter.model.train()
        for ids in np.array_split(rng.permutation(train), batches):
            optimizer.zero_grad(set_to_none=True)
            loss=adapter.loss(adapter.model(adapter.inputs[ids]),ids).mean()
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite training loss")
            loss.backward(); optimizer.step(); scheduler.step(); steps+=1
        if epoch != 1 and epoch % protocol["validation_interval"] and epoch != protocol["epochs"]: continue
        training=adapter.values(train, protocol["batch_size"]); validation_values=adapter.values(validation, protocol["batch_size"])
        score=float(validation_values.mean())
        if not np.isfinite(score): raise FloatingPointError("nonfinite validation error")
        history.append(dict(epoch=epoch, optimizer_steps=steps,
            train_relative_l2=float(training.mean()), val_relative_l2=score,
            val_p95=float(np.percentile(validation_values,95)),
            elapsed_seconds=elapsed+perf_counter()-started,
            learning_rate=optimizer.param_groups[0]["lr"]))
        if score < best:
            best,best_epoch,bad=score,epoch,0; best_state=_model_state(adapter.model)
            _save(directory/"model.pt", dict(state_dict=best_state,best_epoch=best_epoch,config=config))
        else: bad+=1
        _save(resume,dict(specification=specification,model=adapter.model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            epoch=epoch,steps=steps,history=history,best=best,best_epoch=best_epoch,bad=bad,
            elapsed=history[-1]["elapsed_seconds"],best_state=best_state))
        _write_json(directory/"history.json",history)
    if best_state is None: raise RuntimeError("training ended without a validation checkpoint")
    adapter.model.load_state_dict(best_state)
    predictions=np.lib.format.open_memmap(directory/"predictions.npy",mode="w+",dtype=np.complex128,
        shape=(len(test),len(data.coordinates)))
    floor=np.load(_cache_directory(root,data.level,len(train))/"floor_relative_l2.npy",mmap_mode="r")
    rows=[]; cursor=0
    for ids, fields in adapter.predict(test):
        predictions[cursor:cursor+len(ids)] = fields
        target=np.asarray(data.fields[ids]); error2=_norm2(fields-target,data.mass); norm2=_norm2(target,data.mass)
        for i,err,norm in zip(ids,error2,norm2):
            rows.append(dict(query_id=int(i),query_source_id=data.queries[int(i)].get("id",data.queries[int(i)].get("query_id",int(i))),k=float(data.k0[i]),theta=float(data.theta[i]),
                fem_error_pct=float(100*np.sqrt(err/norm)),pod_floor_pct=float(100*floor[i]),
                neural_in_span_error_pct=float(100*np.sqrt(max(err-floor[i]**2*norm,0.)/norm))))
        cursor+=len(ids)
    predictions.flush(); del predictions
    _write_json(directory/"metrics.json",rows)
    errors=np.asarray([row["fem_error_pct"] for row in rows]); floors=np.asarray([row["pod_floor_pct"] for row in rows])
    result=dict(specification=specification,level=data.level,seed=seed,train_count=len(train),parameters=parameters,
        output_modes=adapter.pod.rank,support_dofs=len(data.coordinates),best_epoch=best_epoch,
        epochs_completed=history[-1]["epoch"],stop_reason="validation_patience" if bad>=protocol["patience_checks"] else "epochs_completed",
        training_seconds=history[-1]["elapsed_seconds"],optimizer_steps=steps,
        best_validation_relative_l2=best,test_neural_vs_native_fem_mean_pct=float(errors.mean()),
        test_neural_vs_native_fem_p95_pct=float(np.percentile(errors,95)),
        test_pod_floor_mean_pct=float(floors.mean()),test_pod_floor_rms_pct=float(np.sqrt(np.mean(floors**2))),
        checkpoint_sha256=sha256(directory/"model.pt"),predictions_sha256=sha256(directory/"predictions.npy"),
        metrics_sha256=sha256(directory/"metrics.json"),history_sha256=sha256(directory/"history.json"),
        config_sha256=sha256(directory/"config.json"))
    _write_json(result_path,result)
    return result


def train_all(root=ROOT, protocol=None):
    """Prepare independent level PODs and train all three fresh seeds per mesh."""
    root=Path(root); protocol=_validate_protocol(protocol or {})
    rows,queries_hash=load_queries(root,protocol.get("queries_file","queries.json"))
    declared_hash=protocol.get("primary_queries_sha256" if protocol["train_count"]==1000 else "confirmation_queries_sha256")
    if declared_hash and declared_hash != queries_hash:
        raise ValueError("queries.json hash differs from the frozen protocol")
    expected=(protocol["validation_count"],protocol["test_count"])
    counts=tuple(sum(int(q["split"])==label for q in rows) for label in (1,2))
    train_count=sum(int(q["split"])==0 for q in rows)
    if counts != expected or train_count < protocol["train_count"]:
        raise ValueError(f"query split counts validation/test={counts}, train={train_count} differ from expected {expected} and requested {protocol['train_count']}")
    protocol_hash=json_hash(protocol)
    sources=source_manifest()
    (root/"runs").mkdir(parents=True,exist_ok=True); (root/"results").mkdir(parents=True,exist_ok=True)
    with (root/"runs"/".training.lock").open("a+") as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("another mesh-stability trainer is running") from None
        torch.set_num_threads(protocol["threads"])
        all_results=[]
        for level in LEVELS:
            data=NativeDataset(root,level,rows)
            _projection_cache(root,data,protocol,queries_hash)
            preprocessing_hash=sha256(_cache_directory(root,level,protocol["train_count"])/"preprocessing.json")
            for seed in protocol["seeds"]:
                result=_train_one(root,data,protocol,seed,protocol_hash,queries_hash,preprocessing_hash,sources)
                all_results.append(result)
                _write_json(root/"results"/"training_summary.json",dict(protocol_sha256=protocol_hash,runs=all_results))
        return all_results


def verify(root=ROOT, protocol=None):
    """Replay all nine selected checkpoints and independently rescore native FEM errors."""
    root=Path(root); protocol=_validate_protocol(protocol or {})
    rows,queries_hash=load_queries(root,protocol.get("queries_file","queries.json"))
    declared_hash=protocol.get("primary_queries_sha256" if protocol["train_count"]==1000 else "confirmation_queries_sha256")
    if declared_hash and declared_hash != queries_hash:
        raise ValueError("queries.json hash differs from the frozen protocol")
    protocol_hash=json_hash(protocol); sources=source_manifest(); records=[]
    for level in LEVELS:
        data=NativeDataset(root,level,rows)
        all_train,validation,test=data.split_indices()
        train=all_train[:protocol["train_count"]]
        if len(train)!=protocol["train_count"]: raise ValueError("not enough train queries for verification")
        prep_path=_cache_directory(root,level,len(train))/"preprocessing.json"
        prep_info=json.loads(prep_path.read_text())
        if prep_info["specification"]["train_ids"] != train.tolist():
            raise ValueError("POD training rows differ from shared training split")
        preprocessing_hash=sha256(prep_path)
        for seed in protocol["seeds"]:
            directory=run_directory(root,level,seed,len(train))
            result=json.loads((directory/"result.json").read_text())
            spec=dict(protocol_sha256=protocol_hash,queries_sha256=queries_hash,
                preprocessing_sha256=preprocessing_hash,source_sha256=sources,level=level,
                seed=seed,train_count=len(train),training_ids=train.tolist(),
                validation_ids=validation.tolist(),test_ids=test.tolist())
            if result.get("specification") != spec:
                raise ValueError(f"level {level} seed {seed}: run specification mismatch")
            for name,key in (("model.pt","checkpoint_sha256"),("predictions.npy","predictions_sha256"),
                             ("metrics.json","metrics_sha256"),("history.json","history_sha256"),
                             ("config.json","config_sha256")):
                if sha256(directory/name) != result.get(key):
                    raise ValueError(f"level {level} seed {seed}: changed artifact {name}")
            history=json.loads((directory/"history.json").read_text())
            best=min(history,key=lambda row:row["val_relative_l2"])
            checkpoint=torch.load(directory/"model.pt",map_location="cpu",weights_only=False)
            if best["epoch"] != checkpoint["best_epoch"] or best["epoch"] != result["best_epoch"]:
                raise ValueError("checkpoint was not selected by the recorded validation minimum")
            adapter=Adapter(root,level,seed,protocol)
            adapter.model.load_state_dict(checkpoint["state_dict"])
            saved=np.load(directory/"predictions.npy",mmap_mode="r",allow_pickle=False)
            expected_shape=(len(test),len(data.coordinates))
            if saved.shape != expected_shape or saved.dtype != np.complex128:
                raise ValueError("saved prediction shape/dtype differs from native test fields")
            per_case=[]; offset=0
            for ids,fields in adapter.predict(test):
                block=saved[offset:offset+len(ids)]
                if not np.allclose(fields,block,rtol=3e-5,atol=2e-6):
                    raise ValueError(f"level {level} seed {seed}: checkpoint replay differs from predictions")
                target=np.asarray(data.fields[ids]); norm2=_norm2(target,data.mass)
                err2=_norm2(fields-target,data.mass)
                per_case.extend(100*np.sqrt(err2/norm2)); offset+=len(ids)
            saved_rows=json.loads((directory/"metrics.json").read_text())
            if len(test) != protocol["test_count"] or [row["query_id"] for row in saved_rows] != test.tolist():
                raise ValueError("saved test metrics do not follow shared physical query IDs")
            for row,index in zip(saved_rows,test):
                source_id=data.queries[int(index)].get("id",data.queries[int(index)].get("query_id",int(index)))
                if (row.get("query_source_id",int(index)) != source_id
                        or row["k"] != float(data.k0[index])
                        or row["theta"] != float(data.theta[index])):
                    raise ValueError("saved test frequency/direction differs from shared manifest")
            if not np.allclose(per_case,[row["fem_error_pct"] for row in saved_rows],rtol=1e-6,atol=1e-7):
                raise ValueError("independent native mass metrics differ from saved per-query scores")
            if not np.isclose(np.mean(per_case),result["test_neural_vs_native_fem_mean_pct"],rtol=1e-6,atol=1e-7):
                raise ValueError("independent native test mean differs from run summary")
            records.append(dict(level=level,seed=seed,best_epoch=best["epoch"],test_cases=len(test),
                checkpoint_sha256=result["checkpoint_sha256"],predictions_sha256=result["predictions_sha256"],
                test_mean_relative_l2_pct=float(np.mean(per_case))))
    verification=dict(passed=True,protocol_sha256=protocol_hash,queries_sha256=queries_hash,
                      source_sha256=sources,runs=records)
    _write_json(root/"results"/"verification.json",verification)
    return verification

