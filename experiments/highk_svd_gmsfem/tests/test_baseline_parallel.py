import copy
import numpy as np
import pytest
from experiments.highk_svd_gmsfem.baseline_parallel import merge_worker_results


def worker(i, k, theta):
    protocol = dict(indices=[i], modes=[16, 32], n_coarse=20, fingerprint='frozen')
    common = dict(index=i, k=float(k[i]), theta=float(theta[i]), error_fem_l2=.2, error_percent=20.)
    rows = [dict(common, modes=m, basis_type=2, n_coarse=20, dofs_coarse=21**2*m,
                 fine_degree=2, method=f'GMsFEM CG2 Type2 20x20 M={m}', galerkin_residual=1e-10)
            for m in protocol['modes']]
    return dict(rows=rows, fine_reference_rows=[dict(common, linear_residual=1e-10)],
                metadata=dict(complete=True, protocol=protocol,
                              frequencies=[dict(k=float(k[i]), offline_seconds=1.,
                                                systems=[dict(modes=m) for m in protocol['modes']])]))


def test_merge_requires_complete_disjoint_matching_results():
    k = np.array([18., 20.]); theta = np.array([.1, .2])
    parts = [worker(0, k, theta), worker(1, k, theta)]
    merged = merge_worker_results(parts, [0, 1], k, theta)
    assert len(merged['rows']) == 4
    assert merged['metadata']['unique_offline_seconds'] == 2
    bad = copy.deepcopy(parts); bad[1]['metadata']['protocol']['fingerprint'] = 'changed'
    with pytest.raises(ValueError): merge_worker_results(bad, [0, 1], k, theta)
    bad = copy.deepcopy(parts); bad[1]['rows'].pop()
    with pytest.raises(ValueError): merge_worker_results(bad, [0, 1], k, theta)
    bad = copy.deepcopy(parts); bad[1]['rows'][0]['theta'] += 1
    with pytest.raises(ValueError): merge_worker_results(bad, [0, 1], k, theta)
    with pytest.raises(ValueError): merge_worker_results([parts[0], parts[0]], [0, 1], k, theta)
