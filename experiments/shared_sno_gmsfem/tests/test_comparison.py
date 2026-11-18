"""Cached scores must remain tied to the exact held-out physical queries."""
import numpy as np
import pytest

from experiments.reference_sno_gmsfem.dataset import make_queries
from experiments.reference_sno_gmsfem.report_validation import BASELINE_METHODS, REQUIRED_METRICS
from experiments.shared_sno_gmsfem.comparison import remap_baseline_rows
from experiments.shared_sno_gmsfem.dataset import merge_queries


def fixture():
    queries = {band: make_queries(band, 10, 2) for band in ("low", "high")}
    shared = merge_queries(queries)
    values = {}
    for band, q in queries.items():
        rows = []
        for index in np.flatnonzero(q["split"] == 2):
            for method in BASELINE_METHODS:
                rows.append(dict(index=int(index), sample_index=int(index), method=method,
                                 frequency_id=int(q["frequency_id"][index]),
                                 k=float(q["k0"][index]), theta=float(q["theta"][index]),
                                 **{name: float(index + 1) for name in REQUIRED_METRICS}))
        values[band] = {"rows": rows}
    return queries, shared, values


def test_remapping_preserves_scores_and_all_directions():
    queries, shared, values = fixture()
    rows = remap_baseline_rows(values, queries, shared)
    expected = set(np.flatnonzero(shared["split"] == 2))
    for method in BASELINE_METHODS:
        selected = [r for r in rows if r["method"] == method]
        assert {r["index"] for r in selected} == expected
        at_boundary = [r for r in selected if r["k"] == 2.]
        assert len(at_boundary) == 4
        assert len({r["frequency_id"] for r in at_boundary}) == 1
        for row in selected:
            assert row["sample_index"] == row["index"]
            assert row["scattered_e_complex_ref"] == row["source_index"] + 1


def test_missing_cached_query_is_rejected():
    queries, shared, values = fixture()
    values["low"]["rows"].pop()
    with pytest.raises(ValueError, match="expected all"):
        remap_baseline_rows(values, queries, shared)


@pytest.mark.parametrize("name", ["sample_index", "frequency_id"])
def test_wrong_source_identity_is_rejected(name):
    queries, shared, values = fixture()
    values["high"]["rows"][0][name] += 1
    with pytest.raises(ValueError):
        remap_baseline_rows(values, queries, shared)


def test_changing_a_physical_query_is_rejected():
    queries, shared, values = fixture()
    index = np.flatnonzero(shared["split"] == 2)[0]
    shared["theta"][index] += .1
    with pytest.raises(ValueError, match="physical query differs"):
        remap_baseline_rows(values, queries, shared)
