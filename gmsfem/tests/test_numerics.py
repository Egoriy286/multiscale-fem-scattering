"""Independent physical and algebraic checks; run in DOLFINx Complex."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from gmsfem.fem import assemble_problem, coarse_membership, local_matrices, solve_fine
from gmsfem.geometry import generate_mesh
from gmsfem.offline import build_offline_space, solve_reduced


class ScatteringNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="gmsfem-tests-")
        cls.path = generate_mesh(Path(cls.tmp.name) / "test.msh", n_coarse=4, h=.14)
        cls.problem = assemble_problem(cls.path, case="paper_example1")
        cls.membership = coarse_membership(cls.problem, 4)
        cls.spaces = {
            kind: build_offline_space(cls.problem, n_coarse=4, max_modes=4,
                                      basis_type=kind, progress=None)
            for kind in (1, 2)
        }

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_geometry_area_components_and_coarse_conformity(self):
        p = self.problem
        self.assertAlmostEqual(float(p.areas.sum()), 16., places=10)
        self.assertEqual(set(self.membership), set(range(16)))
        xy = p.coordinates[p.triangles]
        for axis in (0, 1):
            for line in (-1., 0., 1.):
                crosses = (xy[:, :, axis].min(axis=1) < line-1e-9) & (
                    xy[:, :, axis].max(axis=1) > line+1e-9)
                self.assertFalse(np.any(crosses), f"triangle crossing coarse line {axis}={line}")
        included = p.triangles[p.material_tags != 1]
        nodes, inverse = np.unique(included, return_inverse=True)
        tri = inverse.reshape(-1, 3)
        edges = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
        graph = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                                  shape=(len(nodes), len(nodes)))
        self.assertEqual(connected_components(graph, directed=False)[0], 24)
        # Independent counts from the original CAD: 3 small, 16 medium, 5 large.
        for tag, count, radius in ((2, 3, .02), (3, 16, .05), (4, 5, .10)):
            area = p.areas[p.material_tags == tag].sum()
            expected = count*np.pi*radius**2
            # Curvature sizing must resolve even r=.02 at coarse target h.
            # A 32-edge inscribed polygon loses about 0.64% area.
            self.assertLess(abs(area/expected-1), .01)
        ones = np.ones(p.ndofs)
        self.assertAlmostEqual(float(ones @ (p.B @ ones)), 16., places=10)

    def test_local_mass_matrices_against_global_ufl_assembly(self):
        p = self.problem
        mn, sa = [], []
        for block in range(16):
            local = local_matrices(p, np.flatnonzero(self.membership == block))
            for matrix, values in ((local.Mn, mn), (local.Salpha, sa)):
                coo = matrix.tocoo()
                values.append((local.dofs[coo.row], local.dofs[coo.col], coo.data))
        for contributions, expected in ((mn, p.Mn), (sa, p.Salpha)):
            rr, cc, vv = (np.concatenate([item[i] for item in contributions]) for i in range(3))
            assembled = sparse.coo_matrix((vv, (rr, cc)), shape=expected.shape).tocsr()
            relative = sparse.linalg.norm(assembled-expected)/sparse.linalg.norm(expected)
            self.assertLess(relative, 2e-12)

    def test_analytic_total_field_boundary_forcing_identity(self):
        """Tests signs and material interface load using analytic incident field.

        This uses total-field Robin forcing minus a(ui,v), independently of
        the contrast-gradient expression used by assemble_problem.
        """
        import ufl
        from dolfinx import fem
        from dolfinx.fem import petsc as fem_petsc
        from petsc4py import PETSc

        # Oblique incidence and non-unit k also exercise direction handling.
        p = assemble_problem(self.path, case="notebook", k=1.3, direction=(.6, .8))
        x = ufl.SpatialCoordinate(p.mesh)
        ui = ufl.exp(1j*p.k*(.6*x[0]+.8*x[1]))
        v = ufl.TestFunction(p.V)
        normal = ufl.FacetNormal(p.mesh)
        dx = ufl.Measure("dx", domain=p.mesh, metadata={"quadrature_degree": 12})
        ds = ufl.Measure("ds", domain=p.mesh, metadata={"quadrature_degree": 12})
        robin = ufl.dot(ufl.grad(ui), normal)-1j*p.k*ui
        independent_load = (
            ufl.inner(robin, v)*ds
            - ufl.inner(p.alpha_function*ufl.grad(ui), ufl.grad(v))*dx
            + p.k**2*ufl.inner(p.n_function*ui, v)*dx
            + 1j*p.k*ufl.inner(ui, v)*ds
        )
        vector = fem_petsc.assemble_vector(fem.form(independent_load))
        vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        expected = vector.array.copy()
        vector.destroy()
        self.assertLess(np.linalg.norm(expected-p.rhs)/np.linalg.norm(expected), 2e-9)

    def test_homogeneous_medium_has_no_scattered_wave(self):
        p = assemble_problem(self.path, case="homogeneous")
        self.assertEqual(np.linalg.norm(p.rhs), 0.)
        scattered, diagnostics = solve_fine(p)
        self.assertEqual(np.linalg.norm(scattered), 0.)
        self.assertLess(diagnostics["relative_residual"], 1e-12)

    def test_offline_spectra_partition_and_patch_support(self):
        p = self.problem
        for kind, space in self.spaces.items():
            with self.subTest(basis_type=kind):
                eig = space.eigenvalues
                self.assertTrue(np.all(np.isfinite(eig)))
                self.assertGreater(float(eig.min()), -1e-9)
                self.assertTrue(np.all(np.diff(eig, axis=1) >= -1e-9))
                self.assertTrue(np.all(eig[:, 1:] > 0))
                for key in ("max_snapshot_residual", "max_eigen_residual",
                            "chi_dirichlet_residual", "chi_interface_defect"):
                    self.assertLess(space.diagnostics[key], 1e-9, key)
                if kind == 1:
                    np.testing.assert_allclose(np.asarray(space.chi.sum(axis=1)).ravel(), 1., atol=1e-10)
                    mode = space.preview["modes"][:, 0]
                    self.assertLess(np.ptp(mode)/np.max(np.abs(mode)), 1e-8)
                    self.assertLess(np.max(np.abs(eig[:, 0])), 1e-8)
                # Each column is confined to the corresponding vertex patch;
                # on internal patch boundaries it must have zero trace.
                for vertex in range(25):
                    iy, ix = divmod(vertex, 5)
                    lower = np.maximum((-2+ix-1, -2+iy-1), -2)
                    upper = np.minimum((-2+ix+1, -2+iy+1), 2)
                    values = space.P[:, vertex*4:(vertex+1)*4].toarray()
                    outside = np.any((p.coordinates < lower-1e-9) | (p.coordinates > upper+1e-9), axis=1)
                    self.assertLess(np.max(np.abs(values[outside]), initial=0.), 1e-12)
                    for axis in (0, 1):
                        for edge in (lower[axis], upper[axis]):
                            if -2 < edge < 2:
                                trace = np.isclose(p.coordinates[:, axis], edge, atol=1e-9)
                                self.assertLess(np.max(np.abs(values[trace]), initial=0.), 1e-9)

    def test_error_quadrature_mass_norm_phase_and_analytic_total_field(self):
        from gmsfem.metrics import relative_errors

        p = self.problem
        x, y = p.coordinates.T
        reference = (1.2+.3j)+(.4-.2j)*x+(-.1+.6j)*y
        approx = (-.2+.8j)+(.2+.1j)*x+(.3-.2j)*y
        result = relative_errors(p, approx, reference, field="scattered")
        def norm(values):
            return np.sqrt(np.vdot(values, p.M0 @ values).real)
        exact_error = norm(approx-reference)
        self.assertAlmostEqual(result["l2_error_complex"], exact_error, places=11)
        self.assertAlmostEqual(result["e_complex_ref"], 100*exact_error/norm(reference), places=10)
        self.assertAlmostEqual(result["e_complex_paper"], 100*exact_error/norm(approx), places=10)
        phase = relative_errors(p, 1j*reference, reference, field="scattered")
        self.assertLess(phase["l2_error_abs"], 1e-13)
        self.assertAlmostEqual(phase["e_complex_ref"], 100*np.sqrt(2.), places=10)
        # Exact square integral of exp(i*x): 8*sin(2). This tests analytic
        # incident evaluation and both total-field normalization conventions.
        c = 1.+.5j
        ref_constant = np.full(p.ndofs, c)
        approx_constant = 2*ref_constant
        total = relative_errors(p, approx_constant, ref_constant, field="total")
        incident_integral = 8*np.sin(2.)
        def total_norm_squared(value):
            return 16*(abs(value)**2+1)+2*np.real(np.conjugate(value)*incident_integral)
        error = 4*abs(c)
        self.assertAlmostEqual(total["l2_error_complex"], error, places=11)
        self.assertAlmostEqual(total["e_complex_ref"],
                               100*error/np.sqrt(total_norm_squared(c)), places=8)
        self.assertAlmostEqual(total["e_complex_paper"],
                               100*error/np.sqrt(total_norm_squared(2*c)), places=8)

    def test_affine_complex_transfer_on_independent_reference_mesh(self):
        from gmsfem.metrics import P1Evaluator, errors_on_reference

        source = self.problem
        path = generate_mesh(Path(self.tmp.name)/"reference.msh", n_coarse=4, h=.07)
        reference = assemble_problem(path, case=source.case)
        self.assertGreater(reference.ndofs, source.ndofs)
        def affine(points):
            return (1.1-.7j)+(.25+.4j)*points[..., 0]+(-.3+.2j)*points[..., 1]
        source_values = affine(source.coordinates)
        ref_values = affine(reference.coordinates)
        evaluate = P1Evaluator(source, source_values)
        np.testing.assert_allclose(evaluate(reference.coordinates), ref_values, atol=1e-12, rtol=0)
        # Interior points exercise the full (...,2) shape contract, rather
        # than only original vertices and coincident coarse-grid lines.
        rng = np.random.default_rng(402)
        points = rng.uniform(-1.99, 1.99, size=(11, 13, 2))
        np.testing.assert_allclose(evaluate(points), affine(points), atol=1e-12, rtol=0)
        for field in ("scattered", "total"):
            result = errors_on_reference(source, source_values, reference, ref_values,
                                         field=field, chunk_cells=137)
            self.assertLess(result["l2_error_complex"], 1e-12)
            self.assertLess(result["e_complex_ref"], 1e-10)

    def test_enrichment_and_independent_galerkin_residual(self):
        p = self.problem
        fine, _ = solve_fine(p)
        def norm(values):
            return np.sqrt(np.vdot(values, p.M0 @ values).real)
        for kind, space in self.spaces.items():
            errors = []
            for modes in (1, 4):
                solution, _ = solve_reduced(p, space, modes)
                P = space.prolongation(modes)
                residual = P.conjugate().T @ (p.operator @ solution-p.rhs)
                scale = np.linalg.norm(P.conjugate().T @ p.rhs)
                self.assertLess(np.linalg.norm(residual)/scale, 1e-8)
                errors.append(norm(fine-solution)/norm(fine))
            with self.subTest(basis_type=kind, errors=errors):
                self.assertLess(errors[1], errors[0])


if __name__ == "__main__":
    unittest.main()
