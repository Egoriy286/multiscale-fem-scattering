from types import SimpleNamespace

import numpy as np
import torch
import pytest
from scipy.spatial import Delaunay

from neural_operator.data import mass_matrix
from experiments.multiscale_sample_study.fno import (
    FEMGridLoss, FNOAdapter, MPSStencilLoss, model_config, prepare_grid, project_targets,
)
from experiments.mscale_fno_100.training import _bilinear
from experiments.mscale_fno_100.model import MscaleFNO2d
from experiments.multiscale_sample_study.vectorized import VectorizedMscaleFNO2d


def synthetic():
    xx, yy = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))
    points = np.column_stack((xx.ravel(), yy.ravel()))
    triangles = Delaunay(points).simplices
    rng = np.random.default_rng(41)
    return SimpleNamespace(coordinates=points, triangles=triangles,
                           mass_matrix=mass_matrix(points, triangles),
                           u_scat=rng.normal(size=(6, len(points))) + 1j * rng.normal(size=(6, len(points))),
                           A=np.ones((1, len(points))), n=np.ones((1, len(points))),
                           k0=np.linspace(.2, 3, 6), theta=np.zeros(6))


def test_projected_loss_and_gradient_equal_direct_fem():
    data = synthetic()
    prep = prepare_grid(data, [0, 1, 2], resolution=4)
    loss = FEMGridLoss(prep)
    prediction = torch.randn(3, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    ids = [1, 3, 5]
    actual = loss(prediction, ids)
    physical = prediction.flatten(2) * prep['scale'] + torch.as_tensor(prep['mean'])
    T = torch.as_tensor(prep['grid_to_fem'].toarray())
    M = torch.as_tensor(data.mass_matrix.toarray())
    nodes = physical @ T.T
    target = torch.as_tensor(np.stack((data.u_scat[ids].real, data.u_scat[ids].imag), axis=1))
    delta = nodes - target
    expected = (((delta @ M) * delta).sum((1, 2)) / torch.as_tensor(prep['norm2'][ids])).sqrt()
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
    grad_actual, = torch.autograd.grad(actual.sum(), prediction, retain_graph=True)
    grad_expected, = torch.autograd.grad(expected.sum(), prediction)
    torch.testing.assert_close(grad_actual, grad_expected, atol=3e-13, rtol=3e-13)


def test_direct_residual_is_stable_for_representable_fields():
    data = synthetic()
    axis = np.linspace(-1, 1, 4)
    T = _bilinear(data.coordinates, axis, axis)
    rng = np.random.default_rng(7)
    coefficients = rng.normal(size=(2, 16)) + 1j * rng.normal(size=(2, 16))
    fields = (T @ coefficients.T).T
    gram, projected, residual2, norm2 = project_targets(T, data.mass_matrix, fields)
    np.testing.assert_allclose(projected, coefficients, atol=3e-14)
    assert np.all(residual2 >= 0)
    assert np.max(residual2 / norm2) < 1e-28
    assert gram.nnz < 16 ** 2


def test_train_only_statistics_and_preprocessing_reload(tmp_path):
    data = synthetic()
    config = model_config('fno_gelu', width=2, modes=2, depth=1)
    torch.manual_seed(4)
    adapter = FNOAdapter(data, [0, 1, 2], config=config, resolution=4)
    before = {key: np.copy(adapter.prep[key]) for key in ('mean', 'scale', 'k_mean', 'k_std')}
    data.u_scat[3:] *= 1e5
    data.k0[3:] *= 100
    changed = prepare_grid(data, [0, 1, 2], resolution=4)
    for key, value in before.items():
        np.testing.assert_array_equal(changed[key], value)
    adapter.save_preprocessing(tmp_path)
    restored = FNOAdapter.load(tmp_path)
    restored.model.load_state_dict(adapter.model.state_dict())
    np.testing.assert_array_equal(adapter.predict_fields([1, 4]), restored.predict_fields([1, 4]))
    output = adapter.model(adapter.inputs[[1, 4]])
    torch.testing.assert_close(adapter.loss(output, [1, 4]), restored.loss(output, [1, 4]))


def test_nonzero_projection_floor_enters_loss():
    data = synthetic()
    prep = prepare_grid(data, [0, 1, 2], resolution=4)
    optimal = (prep['coefficients'] - prep['mean']) / prep['scale']
    prediction = torch.as_tensor(optimal.reshape(6, 2, 4, 4))
    actual = FEMGridLoss(prep)(prediction, np.arange(6)).numpy()
    np.testing.assert_allclose(actual, prep['floor_relative_l2'], atol=1e-14)
    assert np.all(actual > .1)


def test_vectorized_independent_branches_match_output_and_all_gradients():
    torch.manual_seed(13)
    original = MscaleFNO2d(7, width=2, modes=2, depth=2, scales=(1, 3, 7))
    vectorized = VectorizedMscaleFNO2d(reference=original)
    x = torch.randn(2, 7, 8, 8, requires_grad=True)
    x2 = x.detach().clone().requires_grad_()
    y1, y2 = original(x), vectorized(x2)
    torch.testing.assert_close(y1, y2, atol=2e-6, rtol=2e-5)
    y1.square().sum().backward()
    y2.square().sum().backward()
    torch.testing.assert_close(x.grad, x2.grad, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(original.scales.grad, vectorized.scales.grad, atol=2e-6, rtol=3e-5)
    torch.testing.assert_close(original.gamma.grad, vectorized.gamma.grad, atol=2e-6, rtol=3e-5)
    for branch_index, branch in enumerate(original.branches):
        for name, parameter in branch.named_parameters():
            stacked = vectorized.stacked[name.replace('.', '_DOT_')]
            torch.testing.assert_close(parameter.grad, stacked.grad[branch_index], atol=5e-6, rtol=5e-5)
    restored = VectorizedMscaleFNO2d(7, width=2, modes=2, depth=2, scales=(1, 3, 7))
    restored.load_state_dict(vectorized.state_dict())
    torch.testing.assert_close(restored(x.detach()), vectorized(x.detach()))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')
def test_mps_to_cpu_double_exact_loss_backward():
    prep = prepare_grid(synthetic(), [0, 1, 2], resolution=4)
    prediction = torch.randn(2, 2, 4, 4, device='mps', requires_grad=True)
    value = FEMGridLoss(prep, 'mps')(prediction, [1, 4])
    assert value.device.type == 'cpu' and value.dtype == torch.float64
    value.mean().backward()
    assert prediction.grad.device.type == 'mps'
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize('noise', [0., 1e-4, 1.])
def test_float32_stencil_metric_and_gradients_near_projection_floor(noise):
    prep = prepare_grid(synthetic(), [0, 1, 2], resolution=4)
    normalized = ((prep['coefficients'] - prep['mean']) / prep['scale']).reshape(6, 2, 4, 4)
    x = torch.as_tensor(normalized, dtype=torch.float32) + noise * torch.randn(6, 2, 4, 4)
    x.requires_grad_()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    other = x.detach().to(device).requires_grad_()
    reference = FEMGridLoss(prep)(x, np.arange(6))
    actual = MPSStencilLoss(prep, device)(other, np.arange(6))
    torch.testing.assert_close(reference.float(), actual.cpu(), rtol=1e-5, atol=1e-8)
    reference.mean().backward()
    actual.mean().backward()
    gradient = other.grad.cpu()
    relative_error = torch.linalg.vector_norm(gradient - x.grad) / torch.linalg.vector_norm(x.grad)
    assert relative_error < 1e-5
