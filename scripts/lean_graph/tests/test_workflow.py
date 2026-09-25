import contextlib
import copy
import io
from pathlib import Path
from unittest.mock import patch

from scripts.lean_graph import builds
from scripts.lean_graph.checkpoint import checkpoint
from scripts.lean_graph.evidence import main
from scripts.lean_graph.policy import gate_scope, validate
from scripts.lean_graph.runner import build_context, run_gate
from scripts.lean_graph.snapshot import EvidenceError, capture, digest, file_entry, read_json, write_json
from scripts.lean_graph.tests.test_evidence import EvidenceFixture


class WorkflowTests(EvidenceFixture):
    def add_parent(self):
        parent = copy.deepcopy(self.policy["gates"]["validity"])
        parent["requires"] = ["validity"]
        self.policy["gates"]["parent"] = parent
        self.policy["obligations"]["compiler"]["gates"] = ["parent"]
        return parent

    def scoped_snapshot(self, selected):
        return capture(self.source, self.policy, self.inputs, self.store, gate_scope(self.policy, selected))

    def check(self, manifest, directory):
        with contextlib.redirect_stderr(io.StringIO()):
            return checkpoint("compiler", self.policy, manifest, directory, self.store, self.authority)

    def test_checkpoint_orders_transitive_checks_and_resumes_without_new_runs(self):
        self.add_parent()
        grandparent = copy.deepcopy(self.policy["gates"]["parent"])
        grandparent["requires"] = ["parent"]
        self.policy["gates"]["grandparent"] = grandparent
        self.policy["obligations"]["compiler"]["gates"] = ["grandparent"]
        self.authorize()
        _, manifest, directory = self.scoped_snapshot(["grandparent"])
        first = self.check(manifest, directory)
        self.assertEqual([item["gate"] for item in first["checks"]], ["validity", "parent", "grandparent"])
        self.assertEqual(first["execution"], "passed")
        paths = set((self.store / "runs").iterdir())
        second = self.check(manifest, directory)
        self.assertTrue(all(item["action"] == "reused" for item in second["checks"]))
        self.assertEqual(paths, set((self.store / "runs").iterdir()))

    def test_failed_parent_resumes_matching_predecessor_after_its_input_is_repaired(self):
        parent = self.add_parent()
        fixture = self.root / "parent.json"
        write_json(fixture, {**read_json(self.fixture), "opening": 1})
        self.policy["inputs"]["parent_input"] = {}
        self.inputs["parent_input"] = str(fixture)
        parent["inputs"] = ["parent_input"]
        parent["commands"][0]["argv"][-1] = "{input:parent_input}"
        self.authorize()
        _, manifest, directory = self.scoped_snapshot(["parent"])
        self.assertEqual(self.check(manifest, directory)["execution"], "failed")
        write_json(fixture, read_json(self.fixture))
        _, manifest, directory = self.scoped_snapshot(["parent"])
        retried = self.check(manifest, directory)
        self.assertEqual([(item["gate"], item["action"]) for item in retried["checks"]],
                         [("validity", "reused"), ("parent", "executed")])

    def test_selected_capture_omits_unrelated_projects_but_keeps_identity_and_prerequisites(self):
        self.add_parent()
        self.policy["sources"]["unrelated"] = {"roots": ["missing-unrelated-project"]}
        self.policy["inputs"]["unused"] = {}
        self.inputs["unused"] = str(self.root / "missing-unused-input")
        self.authorize()
        _, manifest, directory = self.scoped_snapshot(["parent"])
        self.assertEqual(set(manifest["sources"]), {"code"})
        self.assertEqual(set(manifest["inputs"]), {"fixture", "package"})
        self.assertFalse((directory / "source/missing-unrelated-project").exists())
        self.assertEqual(self.check(manifest, directory)["execution"], "passed")
        self.package.write_text('["changed identity"]\n')
        _, manifest, directory = self.scoped_snapshot(["parent"])
        self.assertTrue(all(item["action"] == "executed" for item in self.check(manifest, directory)["checks"]))

    def test_cli_explain_separates_diagnostic_execution_from_reviews_and_closure(self):
        policy_file = self.root / "policy.json"
        inputs_file = self.root / "inputs.json"
        write_json(policy_file, self.policy)
        write_json(inputs_file, self.inputs)
        base = ["--source", str(self.source), "--store", str(self.store),
                "--policy", str(policy_file), "--inputs", str(inputs_file)]
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(base + ["explain", "compiler"]), 0)
            self.assertIn("checkpoint compiler", output.getvalue())
            self.assertIn(str(inputs_file), output.getvalue())
            output.seek(0)
            output.truncate()
            self.assertEqual(main(base + ["checkpoint", "compiler"]), 0)
        self.assertIn("execution passed; freshness current; checker diagnostic", output.getvalue())
        self.assertIn("meaning: missing", output.getvalue())
        self.assertIn("Accepted closure: open", output.getvalue())

    def prepare_freshness(self, use="proof"):
        self.policy["graph_gate"] = "metadata"
        self.policy["gates"]["metadata"] = copy.deepcopy(self.policy["gates"]["validity"])
        self.policy["gates"]["metadata"].pop("declaration_freshness", None)
        self.policy["gates"]["validity"]["declaration_freshness"] = {
            "source": "code", "gate": "metadata", "roots": ["Test.closure"], "use": use}
        self.authorize()

    def export(self, manifest, directory, *, meaning="meaning-v1", proof="proof-v1", complete=True):
        record = run_gate("metadata", self.policy, manifest, directory, self.store, self.authority)
        path = next(path for path in (self.store / "runs").glob("*/result.json")
                    if self.authority.read(path) == record)
        write_json(path.parent / "metadata-0.json", [{"root": "Test.closure", "complete": complete,
                   "meaning_key": meaning, "proof_key": proof, "meaning": {}, "proof": {}}])
        record["artifacts"]["metadata-0.json"] = file_entry(path.parent / "metadata-0.json")
        write_json(path, self.authority.sign(record))
        return path

    def test_precise_freshness_needs_current_complete_export_and_preserves_identity_gates(self):
        self.prepare_freshness()
        _, manifest, directory = self.scoped_snapshot(["validity"])
        self.export(manifest, directory)
        run_gate("validity", self.policy, manifest, directory, self.store, self.authority)
        (self.source / "src/Unrelated.lean").write_text("def Unrelated := 9\n")
        _, changed, directory = self.scoped_snapshot(["validity"])
        self.assertFalse(self.status(changed)["gates"]["validity"]["accepted"])
        self.export(changed, directory, complete=False)
        self.assertFalse(self.status(changed)["gates"]["validity"]["accepted"])
        self.export(changed, directory)
        current = self.status(changed)
        self.assertTrue(current["gates"]["validity"]["accepted"])
        self.assertEqual(current["gates"]["validity"]["freshness_basis"], "declarations")
        self.package.write_text('["mandatory identity change"]\n')
        _, changed, directory = self.scoped_snapshot(["validity"])
        self.export(changed, directory)
        self.assertFalse(self.status(changed)["gates"]["validity"]["accepted"])

    def test_changed_proof_and_meaning_keys_reopen_the_declared_checks(self):
        for use in ("proof", "meaning"):
            with self.subTest(use=use):
                self.prepare_freshness(use)
                _, original, directory = self.scoped_snapshot(["validity"])
                self.export(original, directory)
                run_gate("validity", self.policy, original, directory, self.store, self.authority)
                (self.source / "src/Meaning.lean").write_text("changed source " + use)
                _, current, directory = self.scoped_snapshot(["validity"])
                self.export(current, directory, proof="proof-v2")
                self.assertEqual(self.status(current)["gates"]["validity"]["accepted"], use == "meaning")
                self.export(current, directory, meaning="meaning-v2", proof="proof-v3")
                self.assertFalse(self.status(current)["gates"]["validity"]["accepted"])

    def test_tampered_graph_cannot_supply_precise_freshness(self):
        self.prepare_freshness()
        _, old, directory = self.scoped_snapshot(["validity"])
        self.export(old, directory)
        run_gate("validity", self.policy, old, directory, self.store, self.authority)
        (self.source / "src/New.lean").write_text("unrelated")
        _, current, directory = self.scoped_snapshot(["validity"])
        metadata = self.export(current, directory)
        (metadata.parent / "metadata-0.json").write_text("[]")
        result = self.status(current)
        self.assertFalse(result["gates"]["validity"]["accepted"])
        self.assertTrue(result["rejected"])

    def test_inspection_cycle_is_rejected(self):
        self.prepare_freshness()
        self.policy["gates"]["metadata"]["requires"] = ["validity"]
        with self.assertRaises(EvidenceError):
            validate(self.policy)

    def test_managed_build_is_reused_across_gates_and_changed_products_force_rebuild(self):
        self.policy["gates"]["second"] = copy.deepcopy(self.policy["gates"]["validity"])
        self.authorize()
        _, manifest, directory = self.snapshot()
        context = {"dependencies": {"source": digest(manifest)},
                   "command": self.policy["gates"]["validity"]["commands"][0], "runtime": {}, "settings": {}}
        compiled = []

        def execute(command, work, log):
            product = Path(work) / "source/src/.lake/build/checked.olean"
            if not product.exists():
                product.parent.mkdir(parents=True, exist_ok=True)
                product.write_bytes(b"checker-created build result")
                compiled.append(str(work))
            self.assertEqual(product.read_bytes(), b"checker-created build result")
            Path(log).write_text("fixture validity checked\n")
            return {"outcome": "pass", "runtime": {}, "settings": {}}

        with patch("scripts.lean_graph.runner.build_context", return_value=context), \
                patch("scripts.lean_graph.runner.execute", side_effect=execute):
            first = run_gate("validity", self.policy, manifest, directory, self.store, self.authority)
            second = run_gate("second", self.policy, manifest, directory, self.store, self.authority)
            self.assertEqual(len(compiled), 1)
            self.assertEqual(first["commands"][0]["build_cache"]["state"], "miss")
            self.assertEqual(second["commands"][0]["build_cache"]["state"], "hit")
            cached = self.authority.directory / "builds" / digest(context)
            (cached / "products/checked.olean").write_bytes(b"candidate substitution")
            envelope = read_json(cached / "record.json")
            envelope["record"]["files"] = builds.build_files(cached / "products")
            write_json(cached / "record.json", envelope)
            third = run_gate("second", self.policy, manifest, directory, self.store, self.authority)
            self.assertEqual(third["commands"][0]["build_cache"]["state"], "invalid")
            self.assertEqual(len(compiled), 2)

    def test_build_key_changes_with_source_runtime_dependencies_and_options(self):
        command = {"kind": "lean", "cwd": "src", "argv": ["bash", "scripts/validate.sh", "build", "Target"]}
        with patch("scripts.lean_graph.runner.runtime_inputs", return_value={"lean": "version-a"}):
            first = build_context(command, self.root, {"source": "a", "library": "a"}, self.policy)
            for inputs in ({"source": "b", "library": "a"}, {"source": "a", "library": "b"}):
                self.assertNotEqual(digest(first), digest(build_context(command, self.root, inputs, self.policy)))
            changed = copy.deepcopy(command)
            changed["argv"][-1] = "OtherTarget"
            self.assertNotEqual(digest(first), digest(build_context(changed, self.root, first["dependencies"], self.policy)))
        with patch("scripts.lean_graph.runner.runtime_inputs", return_value={"lean": "version-b"}):
            self.assertNotEqual(digest(first), digest(build_context(command, self.root, first["dependencies"], self.policy)))

    def test_failed_gate_cannot_publish_staged_builds(self):
        context = {"command": {"cwd": "src"}, "runtime": {}, "settings": {}}
        self.policy["gates"]["validity"]["commands"].append(
            copy.deepcopy(self.policy["gates"]["validity"]["commands"][0]))
        self.authorize()
        _, manifest, directory = self.snapshot()
        calls = []

        def execute(command, work, log):
            calls.append(command)
            product = Path(work) / "source/src/.lake/build/checked.olean"
            product.parent.mkdir(parents=True, exist_ok=True)
            product.write_bytes(b"build")
            Path(log).write_text("failed later check\n")
            return {"outcome": "pass" if len(calls) == 1 else "timed-out", "runtime": {}, "settings": {}}

        with patch("scripts.lean_graph.runner.build_context", side_effect=[context, None]), \
                patch("scripts.lean_graph.runner.execute", side_effect=execute):
            result = run_gate("validity", self.policy, manifest, directory, self.store, self.authority)
        self.assertEqual(result["outcome"], "timed-out")
        self.assertFalse((self.authority.directory / "builds" / digest(context)).exists())
