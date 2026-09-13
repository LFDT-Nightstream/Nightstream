import re
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.lean_graph.policy import gate_order, gate_scope, load_policy, validate, verify_checker_sources
from scripts.lean_graph.runner import completion
from scripts.lean_graph.snapshot import EvidenceError


class ConformanceRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy()
        self.gates = self.policy["gates"]

    def selected(self, obligation):
        return gate_order(self.policy, self.policy["obligations"][obligation]["gates"])

    def test_build_commands_do_not_drop_required_targets(self):
        # validate.sh's build branch reads only ${2}; later targets are ignored.
        for name, gate in self.gates.items():
            for command in gate["commands"]:
                argv = command["argv"]
                if argv[:3] == ["bash", "scripts/validate.sh", "build"]:
                    with self.subTest(gate=name, argv=argv):
                        self.assertLessEqual(len(argv[3:]), 1)

    def test_pilot_uses_regenerated_current_inputs(self):
        order = self.selected("pilot-conformance")
        for name in ("pilot-result", "pilot-rows"):
            self.assertLess(order.index("pilot-lean-result"), order.index(name))
            data = self.gates[name]["commands"][0]["stdin_json"]
            self.assertEqual(data["package"], "{input:package}")
            self.assertEqual(data["pilot_fixture"], "{input:pilot_fixture}")
        command = self.gates["pilot-lean-result"]["commands"][0]
        self.assertEqual(command["completion"]["equal_files"],
                         [["{output:pilot.json}", "{input:pilot_fixture}"]])
        self.assertIn("pilot-formula-coverage",
                      self.policy["obligations"]["pilot-conformance"]["reviews"])

    def test_child_inputs_and_outputs_cover_every_separate_family(self):
        order = self.selected("piccs-conformance")
        for family in ["K", *[f"A{index}" for index in range(14)]]:
            for side in ("input", "output"):
                name = f"child-{side}-{family.lower()}"
                self.assertIn(name, order)
                command = self.gates[name]["commands"][0]
                data = command["stdin_json"]
                self.assertEqual(data["family"], family)
                self.assertEqual(data["family_result"],
                                 "{input:child_" + side + "_evaluations}/family-" + family + ".json")
                self.assertEqual(data["commitments"], "{input:child_commitments}")
                self.assertEqual(data["folded_cache"], "{input:folded_cache}")
                self.assertLess(order.index("child-commitments"), order.index(name))
                if side == "output":
                    self.assertEqual(data["lean_result"], "{input:recursive_lean_result}")
                    self.assertLess(order.index("recursive-lean-input"), order.index(name))
                else:
                    self.assertNotIn("lean_result", data)
                    self.assertLess(order.index(name), order.index("child-recomposition"))
                self.assertEqual(command["completion"]["tests"],
                                 ["independent_actual_child_evaluation_family"])

    def test_recursive_phase_does_not_reuse_the_base_result(self):
        order = self.selected("piccs-conformance")
        for mode in ("accept", "proof-mutations", "statement-mutations", "output-mutations", "point-mutations"):
            name = "recursive-piccs-" + mode
            self.assertIn(name, order)
            argv = self.gates[name]["commands"][0]["argv"]
            self.assertEqual(argv[-3:], ["{input:recursive_phase_input}", "{input:recursive_lean_result}", mode])
            self.assertLess(order.index("recursive-lean-input"), order.index(name))
        for family in ["k", *[f"a{index}" for index in range(14)], "ccs", "commitment"]:
            command = self.gates["recursive-fresh-" + family]["commands"][0]
            self.assertEqual(command["completion"]["tests"], ["external_positive_fresh_opening_family"])
            self.assertEqual(command["stdin_json"]["cache"], "{input:recursive_opening_cache}")
            self.assertEqual(command["stdin_json"]["lean_result"], "{input:recursive_lean_result}")

    def test_complete_commitment_check_cannot_pass_on_one_row(self):
        gate = self.gates["opening-commitment"]
        command = gate["commands"][0]
        self.assertEqual(command["stdin_json"]["family"], "COMMITMENT")
        test = command["completion"]["tests"][0]
        suffix = f"\ntest {test} ... ok\ntest result: ok. 1 passed; 0 failed;\n"
        completion("independent_commitment_rows=0..22 coefficients=1188 carrier=254260620" + suffix,
                   command["completion"])
        with self.assertRaises(EvidenceError):
            completion("independent_commitment_rows=0..1 coefficients=54 carrier=254260620" + suffix,
                       command["completion"])

    def test_handoffs_and_owner_mutations_are_required(self):
        order = self.selected("piccs-conformance")
        for name in ("child-parent-handoff", "child-recomposition", "recursive-prefix",
                     "candidate-recursive", "candidate-recursive-mutations", "recursive-detached"):
            self.assertIn(name, order)
        data = self.gates["recursive-prefix"]["commands"][0]["stdin_json"]
        self.assertEqual(data["base_fixture"], "{input:recursive_fixture}")
        self.assertEqual(data["ownership"], "{input:piccs_ownership}")
        self.assertEqual(data["phase_input"], "{input:recursive_phase_input}")
        self.assertLess(order.index("piccs-ownership"), order.index("recursive-prefix"))
        command = self.gates["candidate-recursive"]["commands"][0]
        self.assertEqual(command["argv"][-7:], ["{input:expanded}", "{input:recursive_fixture}",
                         "{input:base_fixture}", "{input:phase_input}", "{input:child_running}",
                         "{input:lean_result}", "{input:folded_cache}/folded.json"])

    def test_every_referenced_input_is_captured(self):
        def strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, list):
                for item in value:
                    yield from strings(item)
            elif isinstance(value, dict):
                for item in value.values():
                    yield from strings(item)
        for name, gate in self.gates.items():
            scope = gate_scope(self.policy, [name])
            for value in strings(gate["commands"]):
                for ref in re.findall(r"\{(?:input|value):([^:{}]+)(?::[0-9]+)?\}", value):
                    self.assertIn(ref, scope["inputs"], (name, ref))
            if gate.get("identity_bound"):
                self.assertTrue(set(self.policy["identity_inputs"]) <= set(scope["inputs"]))

    def test_registrations_do_not_grant_review_or_full_step_closure(self):
        for name in ("pilot-assignment", "piccs-assignment", "piccs-public-assignment",
                     "stage1-terminal-assignment", "stage1-terminal-parent"):
            self.assertEqual(set(self.policy["obligations"][name]["reviews"]),
                             {"target-meaning", "decomposition"})
        for name in ("compiler-coverage", "piccs-coverage"):
            self.assertIn("formula-coverage", self.policy["obligations"][name]["reviews"])
        self.assertTrue(self.policy["obligations"]["piccs-coverage"]["coverage"])
        self.assertIsNone(self.policy["obligations"]["stage1-assignment"]["target"])

    def test_compiler_inventory_covers_every_leaf_and_export_connection(self):
        self.assertIn("compiler-declarations", self.selected("compiler-coverage"))
        driver = Path(__file__).resolve().parents[1] / "CompilerCoverage.lean"
        names = re.findall(r"^#audit_axioms (\S+)$", driver.read_text(), re.M)
        self.assertEqual(len(names), len(set(names)))
        leaves = ("StatementBinding", "StatementAbsorption", "ChallengeDerivation",
                  "RoundTranscript", "InitialClaim", "SumcheckChain", "EvalKTerminal",
                  "EvalATerminal", "CcsTerminal", "NormTerminal", "FinalIdentity", "OutputBinding")
        for leaf in leaves:
            prefix = f"NightstreamFPrime.Lifecycle.PiCCS.v1_1.{leaf}."
            for item in ("SpecHolds", "circuit", "soundness", "completeness"):
                self.assertIn(prefix + item, names)
            self.assertTrue(any(name.startswith(prefix + "spec_implies_key") for name in names))
            for item in ("freshColumnCount_eq", "physicalRowCount_eq"):
                self.assertIn(f"NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.{leaf}.{item}", names)
        for item in ("Spec.Folding.PiCCS.accepted_iff_coverage",
                     "Layout.PiCCS.v1_1.physical_implies_phaseHolds",
                     "Layout.PiCCS.v1_1.physical_complete",
                     "Layout.PiCCS.v1_1.cumulativeFootprints_eq_production",
                     "Export.Stage1.PackageCompleteness.complete_piCcsRows",
                     "Export.Stage1.PiCCSDecodedPhase.selectedRowsZero_implies_phaseHolds"):
            self.assertIn("NightstreamFPrime." + item, names)

    def test_compiler_completion_rejects_each_missing_declaration(self):
        driver = Path(__file__).resolve().parents[1] / "CompilerCoverage.lean"
        names = re.findall(r"^#audit_axioms (\S+)$", driver.read_text(), re.M)
        check = self.gates["compiler-declarations"]["commands"][-1]["completion"]
        self.assertEqual(check["patterns"], ["^" + re.escape(name) + r": \[" for name in names])
        lines = [name + ": [propext, Classical.choice, Quot.sound]" for name in names]
        completion("\n".join(lines), check)
        for index, name in enumerate(names):
            with self.subTest(declaration=name), self.assertRaises(EvidenceError):
                completion("\n".join(lines[:index] + lines[index + 1:]), check)

    def test_candidate_cannot_replace_compiler_driver(self):
        relative = Path("scripts/lean_graph/CompilerCoverage.lean")
        original = Path(__file__).resolve().parents[3] / relative
        with TemporaryDirectory() as directory:
            candidate = Path(directory) / "source" / relative
            candidate.parent.mkdir(parents=True)
            candidate.write_bytes(original.read_bytes())
            verify_checker_sources(Path(directory))
            candidate.write_text("-- omitted compiler inventory\n")
            with self.assertRaisesRegex(EvidenceError, "candidate changed the approved checker"):
                verify_checker_sources(Path(directory))

    def test_partial_base_mutation_counts_cannot_satisfy_completion(self):
        for name, label, count in (
            ("proof", "proof_mutations", 2 + 28 * 10 * 2),
            ("statement", "statement-mutations", 4 + 4 + 16 * 17 + 2),
            ("output", "output-mutations", 17 * (17 + 28 + 4) + 10),
        ):
            check = self.gates[f"piccs-{name}-mutations"]["commands"][-1]["completion"]
            prefix = "pi_ccs_complete_phase_values=passed accepted=true elapsed=1\n"
            suffix = " engines=paper_exact,optimized\n"
            complete = f"positive_pi_ccs_{label}_rejected={count}"
            completion(prefix + complete + suffix, check)
            with self.subTest(group=name), self.assertRaises(EvidenceError):
                completion(prefix + f"positive_pi_ccs_{label}_rejected=1" + suffix, check)

    def test_named_branch_cannot_omit_its_check(self):
        policy = deepcopy(self.policy)
        policy["obligations"]["piccs-coverage"].update(
            gates=["piccs-accept"],
            coverage={"proof-mutations": {"scope": "Every round coefficient limb rejects.",
                                           "gates": ["piccs-proof-mutations"]}})
        with self.assertRaisesRegex(EvidenceError, "branch checks are outside the checkpoint"):
            validate(policy)

    def test_named_branches_cover_both_inputs_and_all_mutation_families(self):
        branches = self.policy["obligations"]["piccs-coverage"]["coverage"]
        self.assertEqual(set(branches), {"base-positive", "actual-child-recursive-positive",
                         "complete-phase-result", "proof-mutations", "statement-mutations",
                         "common-prior-point", "output-mutations", "assignment-and-ownership"})
        self.assertEqual(set(self.selected("piccs-coverage")), set(self.selected("piccs-conformance")))
        for group in ("proof", "statement", "output"):
            self.assertEqual(branches[group + "-mutations"]["gates"],
                             [f"piccs-{group}-mutations", f"recursive-piccs-{group}-mutations"])
        self.assertEqual(branches["common-prior-point"]["gates"],
                         ["recursive-piccs-point-mutations", "pilot-result"])
        for family in ["k", *[f"a{index}" for index in range(14)]]:
            self.assertIn("opening-" + family, branches["base-positive"]["gates"])
            for prefix in ("child-input-", "child-output-", "recursive-fresh-"):
                self.assertIn(prefix + family, branches["actual-child-recursive-positive"]["gates"])

    def test_branch_registration_requires_a_scope_and_checks(self):
        for case in ({"scope": "", "gates": ["piccs-accept"]},
                     {"scope": "acceptance", "gates": []},
                     {"scope": "acceptance", "gates": ["unknown-check"]}):
            policy = deepcopy(self.policy)
            policy["obligations"]["piccs-coverage"]["coverage"] = {"case": case}
            with self.subTest(case=case), self.assertRaises(EvidenceError):
                validate(policy)


if __name__ == "__main__":
    unittest.main()
