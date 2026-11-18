"""Physical-frequency grouping and source split preservation in the union."""
import numpy as np
import pytest

from experiments.reference_sno_gmsfem.dataset import make_queries
from experiments.shared_sno_gmsfem.dataset import merge_queries


def sources():
    return {band: make_queries(band) for band in ("low", "high")}


def test_union_preserves_every_query_and_source_split():
    original = sources()
    result = merge_queries(original)
    assert len(result["k0"]) == 2568
    assert len(result["frequencies"]) == 320
    assert np.bincount(result["split"]).tolist() == [1712, 432, 424]
    assert [len(np.unique(result["frequency_id"][result["split"] == j]))
            for j in range(3)] == [214, 54, 52]
    for i, (band, index) in enumerate(zip(result["source_band"], result["source_index"])):
        for name in ("k0", "theta", "split", "medium_id"):
            assert result[name][i] == original[band][name][index]
    boundary = result["k0"] == 2.
    assert boundary.sum() == 16
    assert len(np.unique(result["frequency_id"][boundary])) == 1
    assert np.all(result["split"][boundary] == 2)
    anchor = (result["k0"] == 1.) & (result["theta"] == 0.)
    assert anchor.sum() == 1 and result["split"][anchor].item() == 2


def test_conflicting_splits_at_same_physical_frequency_are_rejected():
    q = sources()
    q["high"]["split"][q["high"]["k0"] == 2.] = 0
    with pytest.raises(ValueError, match="Conflicting source splits at k=2"):
        merge_queries(q)


def test_duplicate_physical_query_is_rejected():
    q = sources()
    q["high"]["theta"][0] = q["low"]["theta"][-8]
    with pytest.raises(ValueError, match="Duplicate physical"):
        merge_queries(q)
