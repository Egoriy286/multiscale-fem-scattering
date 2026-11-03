"""Offline-часть статьи: χ, все граничные snapshots, спектр, sparse prolongation.

Type 1 использует K для гармонических продолжений, Type 2 — K-k²M_n.
Спектральная пара В ОБОИХ случаях: (Ψ.T K Ψ, Ψ.T S_alpha Ψ).
Матрица P хранит базисные функции СТОЛБЦАМИ: u_ms=P U_H.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu

from .fem import FineProblem, coarse_membership, local_matrices, solve_dirichlet


@dataclass
class OfflineSpace:
    P: sparse.csc_matrix
    chi: sparse.csc_matrix
    eigenvalues: np.ndarray
    n_coarse: int
    max_modes: int
    basis_type: int
    build_seconds: float
    diagnostics: dict
    preview: dict

    def prolongation(self, modes: int) -> sparse.csc_matrix:
        """Первые modes мод КАЖДОЙ окрестности (пространства вложены по M)."""
        if isinstance(modes, bool) or int(modes) != modes or not 1 <= modes <= self.max_modes:
            raise ValueError(f"modes должен быть целым от 1 до {self.max_modes}.")
        columns = (np.arange((self.n_coarse + 1) ** 2)[:, None] * self.max_modes + np.arange(modes)).ravel()
        return self.P[:, columns]


def _operator(local, problem: FineProblem, basis_type: int):
    return local.K if basis_type == 1 else local.K - problem.k**2 * local.Mn


def _partition_functions(problem, membership, n_coarse, basis_type):
    """Четыре χ на квадрат: одна факторизация, четыре Q1-граничных следа."""
    H = 4.0 / n_coarse
    rows, columns, entries = [], [], []
    max_residual = 0.0
    for iy in range(n_coarse):
        for ix in range(n_coarse):
            local = local_matrices(problem, np.flatnonzero(membership == iy * n_coarse + ix))
            xy = (problem.coordinates[local.dofs] - (-2 + H * np.array([ix, iy]))) / H
            xi, eta = xy[:, 0], xy[:, 1]
            traces = np.column_stack(((1-xi)*(1-eta), xi*(1-eta), (1-xi)*eta, xi*eta))
            values, residual = solve_dirichlet(_operator(local, problem, basis_type), local.boundary, traces[local.boundary])
            max_residual = max(max_residual, residual)
            vertices = [iy*(n_coarse+1)+ix, iy*(n_coarse+1)+ix+1,
                        (iy+1)*(n_coarse+1)+ix, (iy+1)*(n_coarse+1)+ix+1]
            rows.append(np.repeat(local.dofs, 4))
            columns.append(np.tile(vertices, len(local.dofs)))
            entries.append(values.ravel())

    rows, columns, entries = np.concatenate(rows), np.concatenate(columns), np.concatenate(entries)
    shape = (problem.ndofs, (n_coarse+1)**2)
    total = sparse.coo_matrix((entries, (rows, columns)), shape=shape).tocsc()
    count = sparse.coo_matrix((np.ones_like(entries), (rows, columns)), shape=shape).tocsc()
    reciprocal = count.copy()
    reciprocal.data = 1 / reciprocal.data
    chi = total.multiply(reciprocal).tocsc()
    # Проверить согласованность значений, приходящих с двух сторон coarse edge.
    evaluated = np.asarray(chi.tocsr()[rows, columns]).ravel()
    interface_defect = float(np.max(np.abs(evaluated - entries)))
    if interface_defect > 1e-8:
        raise RuntimeError(f"Функции χ разрывны на крупном ребре: {interface_defect:.3e}")
    chi.eliminate_zeros()
    partition_defect = float(np.max(np.abs(np.asarray(chi.sum(axis=1)).ravel() - 1)))
    if basis_type == 1 and partition_defect > 1e-8:
        raise RuntimeError(f"Type 1 не образует partition of unity: {partition_defect:.3e}")
    # Для Type 2 sum χ ≠ 1 разрешено и ожидаемо из уравнения (15).
    return chi, {"chi_dirichlet_residual": max_residual, "chi_interface_defect": interface_defect,
                 "partition_sum_defect": partition_defect}


def build_offline_space(
    problem: FineProblem,
    *,
    n_coarse: int = 10,
    max_modes: int = 16,
    basis_type: int = 2,
    progress: Callable[[str], None] | None = print,
) -> OfflineSpace:
    """Построить полный offline-базис, независимый от fine-solution и RHS.

    На границе каждого patch задаются ВСЕ дельта-следы; рандомизация и
    извлечение глобального решения здесь не используются. Масштабирование
    столбцов P по L²-норме улучшает обусловленность и не меняет пространство.
    """
    if basis_type not in (1, 2):
        raise ValueError("basis_type должен быть 1 или 2.")
    if isinstance(max_modes, bool) or int(max_modes) != max_modes or max_modes < 1:
        raise ValueError("max_modes должен быть положительным целым.")
    started = perf_counter()
    membership = coarse_membership(problem, n_coarse)
    nvertices = (n_coarse + 1)**2
    if nvertices * max_modes >= problem.ndofs:
        raise ValueError("Крупное пространство не меньше fine. Уменьшите M/n_coarse или уменьшите h.")
    if progress:
        progress(f"Type {basis_type}, {n_coarse}×{n_coarse}: χ на {n_coarse**2} ячейках")
    chi, diagnostics = _partition_functions(problem, membership, n_coarse, basis_type)
    all_rows, all_columns, all_values, all_eigenvalues = [], [], [], []
    snapshot_residuals, eigen_residuals, snapshot_counts = [], [], []
    preview = {}
    preview_vertex = (n_coarse//2)*(n_coarse+1) + n_coarse//2
    tick = max(1, nvertices//5)
    for iy in range(n_coarse+1):
        for ix in range(n_coarse+1):
            vertex = iy*(n_coarse+1)+ix
            coarse_cells = [j*n_coarse+i for j in (iy-1, iy) for i in (ix-1, ix)
                            if 0 <= i < n_coarse and 0 <= j < n_coarse]
            cells = np.flatnonzero(np.isin(membership, coarse_cells))
            local = local_matrices(problem, cells)
            J = len(local.boundary)
            if J < max_modes:
                raise ValueError(f"Patch {vertex}: {J} snapshots < M={max_modes}. Уменьшите h.")
            snapshots, residual = solve_dirichlet(_operator(local, problem, basis_type), local.boundary, np.eye(J))
            A = snapshots.T @ (local.K @ snapshots)
            S = snapshots.T @ (local.Salpha @ snapshots)
            # Только подавление round-off асимметрии, без изменения weight S.
            A, S = (A+A.T)/2, (S+S.T)/2
            try:
                eigenvalues, eigenvectors = linalg.eigh(A, S, subset_by_index=(0, max_modes-1), driver="gvx")
            except linalg.LinAlgError as exc:
                raise RuntimeError(f"Спектральная задача patch {vertex} вырождена. Проверьте fine mesh и локальный резонанс.") from exc
            defect = A @ eigenvectors - (S @ eigenvectors) * eigenvalues
            scale = np.linalg.norm(A) * np.linalg.norm(eigenvectors) + np.linalg.norm((S @ eigenvectors) * eigenvalues)
            eigen_residuals.append(np.linalg.norm(defect) / max(scale, 1e-30))
            modes = snapshots @ eigenvectors
            for j in range(max_modes):
                if modes[np.argmax(np.abs(modes[:, j])), j] < 0:
                    modes[:, j] *= -1
            chi_values = chi[local.dofs, vertex].toarray().ravel()
            basis = chi_values[:, None] * modes
            norms = np.sqrt(np.sum(basis * (problem.M0[local.dofs][:, local.dofs] @ basis), axis=0))
            if np.any(norms < 1e-12) or not np.all(np.isfinite(norms)):
                raise RuntimeError(f"Patch {vertex}: нулевой/неустойчивый столбец P. Уменьшите M или h.")
            basis /= norms
            all_rows.append(np.repeat(local.dofs, max_modes))
            all_columns.append(vertex*max_modes + np.tile(np.arange(max_modes), len(local.dofs)))
            all_values.append(basis.ravel())
            all_eigenvalues.append(eigenvalues)
            snapshot_residuals.append(residual)
            snapshot_counts.append(J)
            if vertex == preview_vertex:
                preview = {"vertex": vertex, "dofs": local.dofs, "cells": cells,
                           "chi": chi_values, "modes": modes[:, :min(4, max_modes)],
                           "basis": basis[:, :min(4, max_modes)], "snapshots": snapshots.shape[1]}
            if progress and ((vertex+1) % tick == 0 or vertex+1 == nvertices):
                progress(f"  snapshots и спектры: {vertex+1}/{nvertices} окрестностей")

    P = sparse.coo_matrix((np.concatenate(all_values), (np.concatenate(all_rows), np.concatenate(all_columns))),
                          shape=(problem.ndofs, nvertices*max_modes)).tocsc()
    P.eliminate_zeros()
    diagnostics.update({"max_snapshot_residual": float(max(snapshot_residuals)),
                        "max_eigen_residual": float(max(eigen_residuals)),
                        "min_snapshots": int(min(snapshot_counts)), "max_snapshots": int(max(snapshot_counts)),
                        "P_nnz": int(P.nnz)})
    if diagnostics["max_eigen_residual"] > 1e-8:
        raise RuntimeError(f"Некорректная спектральная редукция: {diagnostics}")
    return OfflineSpace(P, chi, np.array(all_eigenvalues), n_coarse, max_modes,
                        basis_type, perf_counter()-started, diagnostics, preview)


def solve_reduced(problem: FineProblem, space: OfflineSpace, modes: int) -> tuple[np.ndarray, dict]:
    """Галёркинская проекция и реконструкция; reference solution не используется."""
    P = space.prolongation(modes)
    started = perf_counter()
    PH = P.conjugate().T
    reduced_matrix = (PH @ (problem.operator @ P)).tocsc()
    reduced_rhs = PH @ problem.rhs
    projection_seconds = perf_counter() - started
    started = perf_counter()
    try:
        coefficients = splu(reduced_matrix).solve(reduced_rhs)
    except RuntimeError as exc:
        raise RuntimeError("Крупная система вырождена; проверьте число мод и локальные резонансы.") from exc
    solve_seconds = perf_counter() - started
    solution = np.asarray(P @ coefficients).ravel()
    galerkin_residual = np.linalg.norm(PH @ (problem.operator @ solution - problem.rhs)) / max(np.linalg.norm(reduced_rhs), 1e-30)
    if not np.all(np.isfinite(solution)) or galerkin_residual > 1e-7:
        raise RuntimeError(f"Крупная система не решена: Galerkin residual={galerkin_residual:.3e}")
    return solution, {"dofs_coarse": int(P.shape[1]), "dofs_fine": int(P.shape[0]),
                      "compression": float(P.shape[0]/P.shape[1]),
                      "projection_seconds": projection_seconds, "solve_seconds": solve_seconds,
                      "galerkin_residual": float(galerkin_residual)}
