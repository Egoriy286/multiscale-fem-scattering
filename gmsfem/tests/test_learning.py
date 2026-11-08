"""Saved scientific results must agree with their arrays and CSV exports."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from gmsfem.learning import ROOT, verify_saved
from gmsfem.run import GENERATED_FIGURES, clear_generated_figures


class SavedStudyChecks(unittest.TestCase):
    def test_reused_output_discards_stale_reference_and_cli_figures(self):
        from gmsfem.experiment import save_study

        # A valid tiny study isolates serialization from the already tested PDE solve.
        problem = SimpleNamespace(
            coordinates=np.array([[-2., -2.], [2., -2.], [-2., 2.], [2., 2.]]),
            triangles=np.array([[0, 1, 2], [1, 3, 2]]), material_tags=np.array([1, 1]),
            incident=np.exp(1j * np.array([-2., 2., -2., 2.])), ndofs=4,
            assembly_seconds=0., element_assembly_error=0.)
        row = dict(n_coarse=1, basis_type=1, modes=1, dofs_coarse=4,
                   e_real_paper=0., e_imag_paper=0., e_abs_paper=0., e_complex_ref=0.)
        study = SimpleNamespace(
            problem=problem, fine=np.zeros(4, dtype=complex), rows=[row], solutions={},
            spaces={}, config=dict(case="homogeneous", k=1., direction=[1., 0.], h=.1),
            fine_diagnostics=dict(solve_seconds=0., relative_residual=0.), elapsed_seconds=0.)
        with tempfile.TemporaryDirectory(prefix="gmsfem-reused-") as name:
            output = Path(name)
            for filename in (*GENERATED_FIGURES, "reference_errors.csv", "my_notes.txt", "custom.png"):
                (output / filename).write_text("previous calculation")
            # Saving without reference must clean reference artifacts even outside the CLI.
            save_study(study, output)
            self.assertFalse((output / "reference_errors.csv").exists())
            self.assertFalse((output / "reference_comparison.png").exists())
            clear_generated_figures(output)
            save_study(study, output)
            report = (output / "report.md").read_text()
            self.assertNotIn("![", report)
            self.assertFalse(any((output / f).exists() for f in GENERATED_FIGURES))
            self.assertEqual((output / "my_notes.txt").read_text(), "previous calculation")
            self.assertTrue((output / "custom.png").exists())
            with np.load(output / "fields.npz") as arrays:
                self.assertNotIn("reference_scattered", arrays.files)

    def test_both_preserved_studies_recompute_from_arrays(self):
        for case in ("notebook", "paper_example1"):
            with self.subTest(case=case):
                checked = verify_saved(ROOT / "results" / case)
                self.assertEqual(checked["rows_verified"], 28)
                self.assertEqual(checked["fine_dofs"], 17549)
                self.assertLess(checked["max_exact_P1_error_difference"], 1e-12)

    def test_changed_error_is_rejected(self):
        source = ROOT / "results" / "notebook"
        with tempfile.TemporaryDirectory(prefix="gmsfem-corrupt-") as name:
            directory = Path(name)
            summary = json.loads((source / "summary.json").read_text())
            summary["rows"][0]["e_abs_ref"] += .25
            (directory / "summary.json").write_text(json.dumps(summary))
            for filename in ("fields.npz", "spectra.npz", "errors.csv"):
                (directory / filename).symlink_to(source / filename)
            with self.assertRaises(AssertionError):
                verify_saved(directory)


if __name__ == "__main__":
    unittest.main()
