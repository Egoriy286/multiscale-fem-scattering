"""Stdlib-only, planning-only label ledger. Never imports FEM or training code.

IDs describe query allocation, not generated material realizations. Frequency
values are proposed physical k in [1, 2], not model-normalized features.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

PLAN_VERSION = "1.0"
ROOT = Path(__file__).resolve().parents[2]
CARDINAL = [0, 4, 8, 12]
def _max_gap_order(size=20):
    """Subdivide the largest remaining 1-D grid gap; break ties to the left."""
    selected = [0, size-1]
    while len(selected) < size:
        ordered = sorted(selected)
        gaps = [(b-a, -a, a, b) for a, b in zip(ordered, ordered[1:]) if b-a > 1]
        _, _, left, right = max(gaps)
        selected.append((left+right)//2)
    return selected


FREQUENCY_ORDER = _max_gap_order()


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def parameter_count(width, depth, rank):
    """Exact SNO v2 count; rank=max(rank_A, rank_n, output_rank)."""
    h, l, r = (_positive(width, "width"), _positive(depth, "depth"),
               _positive(rank, "rank"))
    return 8*h + 2 + l*(2*h*h + h + r*r)


def matched_width(target, depth, rank, tolerance=0.01):
    """Nearest positive integer width, with explicit feasibility and tolerance.

    The monotone quadratic gives a finite analytic bound even when rank mixing
    alone exceeds the target. No model or tensors are constructed.
    """
    _positive(target, "target")
    _positive(depth, "depth")
    _positive(rank, "rank")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    minimum = parameter_count(1, depth, rank)
    if target < minimum:
        return {"matched": False, "width": None, "parameters": minimum,
                "relative_error": (minimum-target)/target,
                "reason": "target_below_minimum_positive_width"}
    a, b, c = 2*depth, depth+8, 2+depth*rank*rank-target
    root = (-b + math.sqrt(b*b-4*a*c))/(2*a)
    candidates = {max(1, math.floor(root)), max(1, math.ceil(root))}
    width = min(candidates, key=lambda h: (abs(parameter_count(h, depth, rank)-target), h))
    count = parameter_count(width, depth, rank)
    error = abs(count-target)/target
    return {"matched": error <= tolerance, "width": width,
            "parameters": count, "relative_error": error,
            "reason": "within_tolerance" if error <= tolerance else "integer_width_gap"}


def _group(start, stop, frequencies, directions=CARDINAL):
    return {"medium_start": start, "medium_stop": stop,
            "frequency_ids": list(frequencies), "direction_ids": list(directions)}


def iter_queries(groups):
    """Expand half-open compact medium groups into (medium, frequency, angle)."""
    for group in groups:
        for medium in range(group["medium_start"], group["medium_stop"]):
            for frequency in group["frequency_ids"]:
                for direction in group["direction_ids"]:
                    yield medium, frequency, direction


def query_count(groups):
    return sum((g["medium_stop"]-g["medium_start"])*len(g["frequency_ids"])*
               len(g["direction_ids"]) for g in groups)


def _hashes():
    paths = ["experiments/deep_sno_multimedia/plan.py", "neural_operator/sno/model.py",
             "neural_operator/_layers.py", "neural_operator/datasets/config.py",
             "neural_operator/datasets/generation.py", "neural_operator/data.py",
             "neural_operator/training.py", "experiments/deep_sno_multimedia/protocol.json"]
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
            for p in paths if (ROOT/p).is_file()}


def memory_estimates(samples, media, nodes=3221, cells=6280):
    """Uncompressed array payloads only, excluding solver/optimizer overhead.

    A and n are scalar float64 fields; nodal and DG0 alternatives are listed
    separately. Unique-per-medium storage is the proposed deduplicated layout.
    """
    return {"nodes": nodes, "cells": cells, "samples": samples, "unique_media": media,
            "field_dtype": "complex128", "material_dtype": "float64",
            "u_scat_bytes": samples*nodes*16,
            "two_complex_fields_bytes": samples*nodes*16*2,
            "unique_nodal_A_n_bytes": media*nodes*8*2,
            "unique_DG0_A_n_bytes": media*cells*8*2,
            "duplicated_per_query_nodal_A_n_bytes": samples*nodes*8*2,
            "duplicated_per_query_DG0_A_n_bytes": samples*cells*8*2,
            "unique_nodal_and_DG0_A_n_bytes": media*(nodes+cells)*8*2,
            "scope": "coarse array payload only; nodal and DG0 may coexist; QA bytes excluded"}


def build_plan(branch="frequency"):
    if branch not in ("frequency", "diversity"):
        raise ValueError("branch must be frequency or diversity")
    f5, f10, f20 = list(range(5)), list(range(10)), list(range(20))
    scout = [_group(0, 20, f20), _group(20, 40, f10), _group(40, 80, f5)]
    completion = ([_group(20, 40, range(10, 20)), _group(40, 80, range(5, 20)), _group(80, 90, f20)]
                  if branch == "frequency" else [_group(80, 280, f5)])
    splits = {"train": scout + completion,
              "validation": [_group(10000, 10020, f5),
                             _group(10000, 10020, [20, 21, 22]),
                             _group(10000, 10020, [0, 1], [1, 5, 9, 13])],
              "test": [_group(20000, 20040, f5),
                       _group(20000, 20040, [23, 24, 25]),
                       _group(20000, 20040, [0, 1], [2, 6, 10, 14])],
              "qa": [_group(30000, 30020, f5)]}
    for split in ("validation", "test"):
        for group, cohort in zip(splits[split], ("shared_k_base_theta", "reserved_k_base_theta", "shared_k_shifted_theta")):
            group["cohort"] = cohort
    counts = {name: query_count(groups) for name, groups in splits.items()}
    all_media = sum(len({m for m, _, _ in iter_queries(groups)})
                    for name, groups in splits.items() if name != "qa")
    target = parameter_count(64, 12, 64)
    return {"plan_version": PLAN_VERSION, "planning_only": True,
            "dataset_generated": False, "branch": branch,
            "query_key": ["medium_id", "frequency_id", "direction_id"],
            "scope": "fixed CAD geometry, new independent material identities; no sampler implemented",
            "coarse_solves": {"train_LU": 1800 if branch == "frequency" else 800+1000, "validation_LU": 160, "test_LU": 320, "total_coarse_LU": 2280, "coarse_RHS": 9600, "qa_cost": "unspecified refinement protocol; not counted as coarse solves"},
            "test_protocol": "new materials in both holdouts; five shared k by four base angles; three reserved interlaced k by four base angles; two shared k by four shifted angles (validation pi/8, test pi/4); angular probe conditional on new materials",
            "identity_policy": "disjoint medium IDs across splits and QA case domain; no material sampling performed",
            "candidate_train_medium_range": [0, 280],
            "medium_ranges_half_open": {"train_base": [0, 80],
                "train_diversity_extension": [80, 280], "validation": [10000, 10020],
                "test": [20000, 20040], "qa": [30000, 30020]},
            "qa_protocol": {"domain": "qa_reserved_cases", "protocol_id": "qa-v1",
                "case_ids": list(range(30000, 30020)), "counts_against_global_budget": True,
                "allocation_state": "reserved_budget_slots_not_physical_FEM_queries",
                "requires_mesh_id_in_executable_query_key": True},
            "frequency_table": ([{"frequency_id": i, "k": 1+j/19, "domain": "train_shared"}
                                for i, j in enumerate(FREQUENCY_ORDER)] +
                                [{"frequency_id": 20+i, "k": 1+(j+.5)/19,
                                  "domain": "validation_interlaced" if i < 3 else "test_interlaced"}
                                 for i, j in enumerate((1, 7, 13, 3, 9, 15))]),
            "frequency_policy": "proposed physical k in [1,2]; six disjoint interlaced holdout k reserved; F5/F10/F20 nested farthest-point max-gap grid ordering",
            "direction_table": [{"direction_id": d, "theta_radians": 2*math.pi*d/16}
                                for d in range(16)],
            "scouts": {"A": [_group(0, 20, f20)], "B": [_group(0, 40, f10)],
                       "C": [_group(0, 80, f5)]},
            "scout_union": scout, "completion": completion, "splits": splits,
            "counts": dict(counts, scout_union=3200, completion=4000, global_total=sum(counts.values())),
            "budget": {"global": 10000, "train_cap": 7200, "validation": 800,
                       "test": 1600, "qa": 400},
            "memory": dict(memory_estimates(sum(v for k, v in counts.items() if k != "qa"), all_media), qa_mesh_dependent_bytes=None, train_dense_POD_component_float64_bytes=7200*3221*8),
            "architecture": {"version": 2, "parameter_formula": "8H+2+L*(2H^2+H+R^2)",
                "rank_policy": "R=max(rank_A,rank_n,output_rank)",
                "conditional_example": {"rank": 64, "reference_width": 64,
                    "reference_depth": 12, "target_parameters": target, "tolerance": .01,
                    "depth_matches": {str(l): matched_width(target, l, 64) for l in (3, 6, 12)}}},
            "source_sha256": _hashes()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("frequency", "diversity"), default="frequency")
    parser.add_argument("--output", type=Path, help="explicit metadata JSON destination")
    args = parser.parse_args(argv)
    plan = build_plan(args.branch)
    if args.output is not None:
        args.output.write_text(json.dumps(plan, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"plan_version": plan["plan_version"], "planning_only": True,
                      "branch": args.branch, "counts": plan["counts"],
                      "memory": plan["memory"]}, indent=2))


if __name__ == "__main__":
    main()
