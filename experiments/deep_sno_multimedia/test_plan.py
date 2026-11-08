"""Planning ledger checks only: no FEM solves, training, or dataset writes."""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unittest

_SPEC = importlib.util.spec_from_file_location("multimedia_plan", Path(__file__).with_name("plan.py"))
plan = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(plan)


class PlanningTests(unittest.TestCase):
    def test_protocol_and_scenario_ledgers_agree(self):
        protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
        self.assertEqual(protocol["status"], "planned_not_executed")
        self.assertEqual(protocol["execution_state"]["fem_fields_generated"], 0)
        self.assertEqual(protocol["execution_state"]["models_trained"], 0)
        for branch in ("frequency", "diversity"):
            ledger = plan.build_plan(branch)
            for name in ("train", "validation", "test"):
                self.assertEqual(protocol["budget"][name], ledger["counts"][name])
            self.assertEqual(protocol["budget"]["qa_cap"], ledger["counts"]["qa"])
            self.assertEqual(protocol["budget"]["total_cap"], ledger["counts"]["global_total"])
            self.assertTrue(ledger["qa_protocol"]["requires_mesh_id_in_executable_query_key"])
            self.assertEqual(ledger["qa_protocol"]["allocation_state"],
                             "reserved_budget_slots_not_physical_FEM_queries")
            for arm, groups in zip(protocol["scout"]["arms"], ledger["scouts"].values()):
                self.assertEqual(arm["fields"], plan.query_count(groups))
            self.assertEqual(protocol["depth_study"]["depths"],
                             list(map(int, ledger["architecture"]["conditional_example"]["depth_matches"])))

    def test_exact_budgets_and_no_query_or_medium_leakage(self):
        for branch in ("frequency", "diversity"):
            p = plan.build_plan(branch)
            seen_queries, seen_media = set(), set()
            for split, groups in p["splits"].items():
                queries = list(plan.iter_queries(groups))
                unique = set(queries)
                self.assertEqual(len(queries), len(unique))
                self.assertEqual(len(queries), p["budget"].get(split, 7200))
                media = {q[0] for q in unique}
                self.assertFalse(seen_queries & unique)
                self.assertFalse(seen_media & media)
                seen_queries.update(unique)
                seen_media.update(media)
                if split != "train":
                    self.assertTrue(all(m >= 10000 for m in media))
            self.assertEqual(len(seen_queries), 10000)
            self.assertEqual(p["counts"]["global_total"], 10000)

    def test_scouts_nested_and_reused_once(self):
        for branch in ("frequency", "diversity"):
            p = plan.build_plan(branch)
            scouts = [set(plan.iter_queries(p["scouts"][key])) for key in "ABC"]
            self.assertEqual([len(s) for s in scouts], [1600]*3)
            union = set.union(*scouts)
            self.assertEqual(len(union), 3200)
            self.assertEqual(union, set(plan.iter_queries(p["scout_union"])))
            self.assertEqual([{q[0] for q in s} for s in scouts],
                             [set(range(n)) for n in (20, 40, 80)])
            freqs = [{q[1] for q in s} for s in scouts]
            self.assertTrue(freqs[2] < freqs[1] < freqs[0])
            completion = set(plan.iter_queries(p["completion"]))
            self.assertEqual(len(completion), 4000)
            self.assertFalse(union & completion)
            self.assertEqual(union | completion, set(plan.iter_queries(p["splits"]["train"])))

    def test_branch_caps(self):
        for branch, media_count, frequency_count in (("frequency", 90, 20), ("diversity", 280, 20)):
            p = plan.build_plan(branch)
            queries = set(plan.iter_queries(p["splits"]["train"]))
            self.assertEqual(len({m for m, _, _ in queries}), media_count)
            self.assertEqual(len({f for _, f, _ in queries}), frequency_count)
            if branch == "frequency":
                self.assertEqual(queries, {(m, f, d) for m in range(90) for f in range(20) for d in plan.CARDINAL})
            else:
                extra = {(m, f, d) for m in range(80, 280) for f in range(5) for d in plan.CARDINAL}
                self.assertTrue(extra <= queries)

    def test_test_queries_and_physical_angles(self):
        p = plan.build_plan()
        angles = {d["direction_id"]: d["theta_radians"] for d in p["direction_table"]}
        self.assertEqual(len(set(angles.values())), 16)
        for j, d in enumerate(plan.CARDINAL):
            self.assertAlmostEqual(angles[d], 2*math.pi*j/4)
        groups = p["splits"]["test"]
        self.assertEqual([plan.query_count([g]) for g in groups], [800, 480, 320])
        self.assertEqual(groups[0]["frequency_ids"], list(range(5)))
        self.assertEqual(groups[1]["frequency_ids"], [23, 24, 25])
        self.assertEqual(groups[2]["direction_ids"], [2, 6, 10, 14])

        train_f = {q[1] for q in plan.iter_queries(p["splits"]["train"])}
        val_reserved = set(p["splits"]["validation"][1]["frequency_ids"])
        test_reserved = set(groups[1]["frequency_ids"])
        self.assertFalse(train_f & val_reserved or train_f & test_reserved or val_reserved & test_reserved)
        values = [row["k"] for row in p["frequency_table"]]
        self.assertEqual(len(values), len(set(values)))
        self.assertEqual(p["coarse_solves"]["total_coarse_LU"], 2280)
        for split, phase in (("validation", math.pi/8), ("test", math.pi/4)):
            shifted = p["splits"][split][2]
            for j, direction in enumerate(shifted["direction_ids"]):
                self.assertAlmostEqual(angles[direction], phase+2*math.pi*j/4)
            self.assertTrue(set(shifted["frequency_ids"]) <= set(range(5)))

    def test_schema_hashes_memory_and_frequency_nesting(self):
        p = plan.build_plan()
        self.assertTrue(p["planning_only"])
        self.assertFalse(p["dataset_generated"])
        self.assertEqual(p["query_key"], ["medium_id", "frequency_id", "direction_id"])
        self.assertEqual(p["qa_protocol"]["domain"], "qa_reserved_cases")
        self.assertEqual(len(p["frequency_table"]), 26)
        f5 = [row["k"] for row in p["frequency_table"][:5]]
        self.assertEqual((min(f5), max(f5)), (1, 2))
        for source, digest in p["source_sha256"].items():
            self.assertEqual(digest, hashlib.sha256((plan.ROOT/source).read_bytes()).hexdigest())
        m = p["memory"]
        self.assertEqual(m["u_scat_bytes"], 9600*3221*16)
        self.assertEqual(m["unique_nodal_A_n_bytes"], 150*3221*16)
        self.assertEqual(plan.build_plan("diversity")["memory"]["unique_media"], 340)

    def test_parameter_count_matches_source_linear_definitions(self):
        def linear_assignments(relative, class_name):
            module = ast.parse((plan.ROOT/relative).read_text())
            cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == class_name)
            init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
            found = []
            for node in ast.walk(init):
                if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Attribute)
                        and isinstance(node.value.func.value, ast.Name)
                        and node.value.func.value.id == "nn" and node.value.func.attr == "Linear"):
                    found.append((ast.unparse(node.targets[0]),
                                  tuple(ast.unparse(a) for a in node.value.args),
                                  {k.arg: ast.literal_eval(k.value) for k in node.value.keywords}))
            return found
        model = linear_assignments("neural_operator/sno/model.py", "SVDSNO")
        block = linear_assignments("neural_operator/_layers.py", "CoefficientBlock")
        self.assertIn(("self.lift", ("5", "self.width"), {}), model)
        self.assertIn(("self.project", ("width", "2"), {}), model)
        self.assertEqual(set((name, args, tuple(kwargs.items())) for name, args, kwargs in block),
                         {("self.local", ("width", "width"), ()),
                          ("self.channels", ("width", "width"), (("bias", False),)),
                          ("self.modes", ("modes", "modes"), (("bias", False),))})
        # Independent count from the verified Linear layers: weights + biases.
        h, r, depth = 64, 64, 12
        source_count = (5*h+h) + (h*2+2) + depth*((h*h+h)+(h*h)+(r*r))
        self.assertEqual(source_count, plan.parameter_count(h, depth, r))
        architecture = plan.build_plan()["architecture"]["conditional_example"]
        self.assertEqual(architecture["reference_depth"], depth)
        self.assertEqual(set(architecture["depth_matches"]), {"3", "6", "12"})
        self.assertEqual(architecture["target_parameters"], source_count)

    def test_parameter_formula_and_finite_matching(self):
        target = plan.parameter_count(64, 12, 64)
        self.assertEqual(target, 148738)
        self.assertEqual(plan.matched_width(target, 12, 64)["width"], 64)
        for depth in (3, 6, 12):
            match = plan.matched_width(target, depth, 64)
            if match["width"] is not None:
                h = match["width"]
                self.assertEqual(match["parameters"], plan.parameter_count(h, depth, 64))
                for candidate in (max(1, h-1), h+1):
                    self.assertLessEqual(abs(match["parameters"]-target),
                                         abs(plan.parameter_count(candidate, depth, 64)-target))
                self.assertEqual(match["matched"], match["relative_error"] <= .01)
        impossible = plan.matched_width(10, 100, 10000)
        self.assertFalse(impossible["matched"])
        self.assertIsNone(impossible["width"])
        self.assertFalse(plan.matched_width(target+1, 12, 64, tolerance=0)["matched"])
        with self.assertRaises(ValueError):
            plan.build_plan("unknown")
        with self.assertRaises(ValueError):
            plan.matched_width(target, 0, 64)


if __name__ == "__main__":
    unittest.main()
