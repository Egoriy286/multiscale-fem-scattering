from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from neural_operator.mesh_study import (material_table,apply_materials,
    select_heldout_cases,relative_fem_error,run_cross_mesh_study)
from neural_operator.data import mass_matrix


def test_materials_follow_unique_physical_tags():
    table = material_table(np.array([1,2,2,3]),np.array([1.,.6,.6,.9]),np.array([1.,3.,3.,2.]))
    a,n = apply_materials(np.array([3,1,2,3]),table)
    assert np.array_equal(a,[.9,1.,.6,.9])
    assert np.array_equal(n,[2.,1.,3.,2.])
    with pytest.raises(ValueError,match='constant'):
        material_table(np.array([2,2]),np.array([.6,.7]),np.ones(2))
    with pytest.raises(ValueError,match='unknown'):
        apply_materials(np.array([4]),table)


def test_heldout_frequency_selection():
    ids = np.repeat(np.arange(10),8)
    split = np.repeat([0,0,0,0,0,1,1,2,2,2],8)
    k = np.repeat(np.linspace(1,2,10),8)
    theta = np.tile(np.arange(8)*np.pi/4,10)
    indices = select_heldout_cases(ids,split,k,theta)
    assert np.array_equal(indices,[56,58,64,66,72,74])
    assert np.all(split[indices] == 2)
    with pytest.raises(ValueError,match='frequency'):
        select_heldout_cases(ids,split,k,theta,n_frequencies=4)


def test_consistent_error_zero_semantics():
    coords = np.array([[0.,0.],[1.,0.],[0.,1.]])
    mass = mass_matrix(coords,np.array([[0,1,2]]))
    true = np.array([[1j,2j,3j],[0j,0j,0j]])
    assert relative_fem_error(true,true,mass).tolist() == [0.,0.]
    assert np.isinf(relative_fem_error(true[1],np.ones(3),mass))
    assert relative_fem_error(true[:1],true[:1]*1.5,mass)[0] == pytest.approx(.5)


def test_study_artifacts_and_frozen_weights_without_loading_labels(tmp_path,monkeypatch):
    """Small analytic fake solver exercises orchestration without model fitting."""
    import neural_operator.mesh_study as study
    import neural_operator.training as training
    xy = np.array([[-2.,-2.],[2.,-2.],[2.,2.],[-2.,2.]])
    triangles = np.array([[0,1,2],[0,2,3]])
    target_xy = np.vstack((xy,[[0.,0.]]))
    target_triangles = np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
    def analytic(coords,k,theta):
        return k[:,None]*(1+coords[:,0][None,:])+1j*(theta[:,None]+coords[:,1][None,:])
    class Solver:
        created = []
        solved = 0
        def __init__(self,path):
            self.created.append(Path(path))
            target = Path(path).parent.name == 'tiny'
            self.coordinates = target_xy if target else xy
            self.triangles = target_triangles if target else triangles
            self.cell_tags = np.ones(len(self.triangles),dtype=int)
        def nodal_coefficients(self,cells):
            return np.full(len(self.coordinates),cells[0])
        def solve(self,a,n,k,theta):
            Solver.solved += 1
            return analytic(self.coordinates,np.array([k]),np.array([theta]))[0]
    class Predictor:
        coordinates = xy
        bases = {name:SimpleNamespace(rank=0) for name in ('A','n')}
        def __init__(self):
            self.triangles = triangles
        def predict(self,A,n,theta,k):
            return .95*analytic(self.coordinates,np.asarray(k),np.asarray(theta))
        def on_mesh(self,coordinates,connectivity):
            return SimpleNamespace(predict_static=lambda theta,k: .95*analytic(coordinates,np.asarray(k),np.asarray(theta)))
    monkeypatch.setattr(study,'HelmholtzSolver',Solver)
    monkeypatch.setattr(study,'generate_mesh',lambda path,**kw:path)
    monkeypatch.setattr(training,'load_predictor',lambda path:Predictor())
    directory = tmp_path/'dataset'
    directory.mkdir()
    np.savez_compressed(directory/'geometry.npz',dof_coords=xy,triangles=triangles,cell_tags=np.ones(2,dtype=int))
    # Deliberately omit u_scat: the study must re-solve, never load labels.
    np.savez_compressed(directory/'fem_dataset.npz',A_cells=np.ones((1,2)),n_cells=np.ones((1,2)),k0=np.repeat(np.linspace(1,2,6),4),theta=np.tile(np.arange(4)*np.pi/2,6),frequency_id=np.repeat(np.arange(6),4),split=np.repeat([0,0,0,2,2,2],4),split_strategy=np.asarray('frequency'))
    checkpoint = tmp_path/'model.pt'
    checkpoint.write_bytes(b'frozen checkpoint')
    rows,summaries,fields,info = run_cross_mesh_study({'low':checkpoint},{'low':directory},tmp_path/'study',mesh_specs={'tiny':dict(h=.3,n_coarse=2)})
    assert len(rows) == 12 and len(summaries) == 2
    assert len(Solver.created) == 2  # One target mesh reused; one source mesh.
    assert Solver.solved == 12  # Permutation uses no additional reference solves.
    assert all(row['error_model_vs_target_fem'] == pytest.approx(.05) for row in rows)
    assert max(row['error_source_fem_vs_target_fem'] for row in rows) < 1e-14
    assert max(row['commutation_error_relative'] for row in rows) < 1e-14
    assert fields['low__tiny__reference'].shape == (6,5)
    assert set(info) == {'tiny','permuted'}
    metadata = json.loads(fields['metadata_json'].item())
    assert metadata['Ns'] == 6
    assert metadata['training_epochs'] == 0 and metadata['refit'] is False
    queries = metadata['selected_queries']['low']
    assert len(queries['sample_indices']) == len(queries['frequency_ids']) == len(queries['k0']) == len(queries['theta']) == 6
    assert queries['k0'][-1] == 2.
    assert np.array_equal(fields['low__tiny__sample_indices'],queries['sample_indices'])
    assert len(metadata['source_geometry']['low']['coordinates_sha256']) == 64
    assert len(metadata['source_geometry']['low']['triangles_sha256']) == 64
    for mesh in info.values():
        assert len(mesh['coordinates_sha256']) == len(mesh['triangles_sha256']) == 64
        assert mesh['coordinates_dtype'] == str(xy.dtype)
        assert mesh['training_epochs'] == 0 and mesh['refit'] is False
        for mapping in mesh['mapping_by_band']['low'].values():
            assert mapping['coverage'] == 1.
            assert mapping['max_abs_rowsum_minus_one'] < 1e-14
            assert 1 <= mapping['max_nnz_row'] <= 3
    assert metadata['checkpoints']['low']['unchanged']
    assert metadata['checkpoints']['low']['before'] == metadata['checkpoints']['low']['after']
    for name in ('per_case.csv','summary.json','fields.npz','mesh_info.json'):
        assert (tmp_path/'study'/name).exists()
    assert checkpoint.read_bytes() == b'frozen checkpoint'
