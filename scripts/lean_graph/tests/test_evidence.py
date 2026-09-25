from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.lean_graph.evidence import main
from scripts.lean_graph import guard
from scripts.lean_graph.policy import Authority, CAPS, checker_key, validate
from scripts.lean_graph.records import changes, markdown, report
from scripts.lean_graph.runner import build_lock, completion, execute, run_gate
from scripts.lean_graph.snapshot import (EvidenceError, capture, digest, entries, inspect,
                                     read_json, verify, write_json)


FIXTURES = Path(__file__).with_name("fixtures")


class EvidenceFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        lock = patch.object(guard, "_LOCK_PATH", self.root / "build.lock")
        lock.start()
        self.addCleanup(lock.stop)
        self.source, self.store = self.root / "candidate", self.root / "evidence"
        (self.source / "src").mkdir(parents=True)
        shutil.copyfile(FIXTURES / "gate.py", self.source / "src/check.py")
        (self.source / "src/Meaning.lean").write_text("def Meaning := 2\n")
        self.fixture = self.root / "fixture.json"
        shutil.copyfile(FIXTURES / "zero.json", self.fixture)
        self.package = self.root / "package.json"
        self.package.write_text('["preserved package A"]\n')
        self.inputs = {"fixture": str(self.fixture), "package": str(self.package)}
        command = {"kind": "python", "cwd": "src",
                   "argv": [sys.executable, "check.py", "{input:fixture}"],
                   "completion": {"patterns": ["^fixture validity checked$"],
                                  "tests": ["required_case"]}}
        self.policy = {
            "schema": 1, "sources": {"code": {"roots": ["src"], "exclude": ["*.md"]}},
            "inputs": {"fixture": {}, "package": {}}, "identity_inputs": ["package"],
            "gates": {"validity": {"sources": ["code"], "inputs": ["fixture"],
                                    "identity_bound": True, "commands": [command]}},
            "reviews": {"meaning": {"scope": "exact registered target"}},
            "obligations": {
                "compiler": {"owner": "owner criterion", "status": "Compiler-closed",
                             "target_required": True, "target": "Test.Target",
                             "gates": ["validity"], "reviews": ["meaning"], "gap": "Prove Target."},
                "conformance": {"owner": "owner conformance", "status": "Conformance-closed",
                                "gates": ["validity"], "reviews": ["meaning"], "gap": "Check input."},
                "production": {"owner": "owner production", "status": "Production-closed",
                               "gates": [], "reviews": [], "gap": "Production verifier is not selected."}}}
        self.authority_path = self.root / "checker"
        self.authority_path.mkdir()
        self.authorize()

    def authorize(self):
        validate(self.policy)
        write_json(self.authority_path / "policy.json", self.policy)
        write_json(self.authority_path / "approval.json", {
            "outcome": "pass", "reviewer": "test owner", "policy": digest(self.policy),
            "checker": checker_key()})
        (self.authority_path / "record.key").write_bytes(b"test fixture authentication key")
        self.authority = Authority(self.authority_path)

    def snapshot(self):
        return capture(self.source, self.policy, self.inputs, self.store)

    def run_validity(self, authority=True):
        identity, manifest, directory = self.snapshot()
        result = run_gate("validity", self.policy, manifest, directory, self.store,
                          self.authority if authority else None)
        return result, manifest

    def review(self, manifest, **overrides):
        record = {"review": "meaning", "scope": "exact registered target", "reviewer": "independent reviewer",
                  "outcome": "pass", "snapshot": digest(manifest), "policy": digest(self.policy)}
        record.update(overrides)
        write_json(self.authority_path / "reviews/meaning.json", self.authority.sign(record))

    def status(self, manifest):
        return report(self.policy, manifest, self.store, self.authority)


class EvidenceTests(EvidenceFixture):
    def test_real_process_checks_zero_fixture_without_claiming_branch_coverage(self):
        result, manifest = self.run_validity()
        self.assertEqual(result["outcome"], "pass")
        self.assertFalse(self.status(manifest)["statuses"]["Compiler-closed"])
        self.review(manifest)
        result = self.status(manifest)
        self.assertTrue(result["statuses"]["Compiler-closed"])
        self.assertTrue(result["statuses"]["Conformance-closed"])
        self.assertFalse(result["statuses"]["Production-closed"])

    def test_nonzero_opening_with_zero_matrix_evaluation_is_valid(self):
        shutil.copyfile(FIXTURES / "nonzero.json", self.fixture)
        result, _ = self.run_validity()
        self.assertEqual(result["outcome"], "pass")

    def test_invalid_bound_commitment_public_and_evaluation_fail(self):
        original = read_json(self.fixture)
        for field in ("opening", "commitment", "public", "evaluation"):
            with self.subTest(field=field):
                write_json(self.fixture, {**original, field: 1})
                result, _ = self.run_validity()
                self.assertEqual(result["outcome"], "failed")

    def test_definition_change_invalidates_evidence_but_old_snapshot_survives(self):
        record, old = self.run_validity()
        (self.source / "src/Meaning.lean").write_text("def Meaning := 3\n")
        _, current, _ = self.snapshot()
        self.assertIn("source:code", changes(record, current, self.policy["gates"]["validity"], self.policy))
        self.assertEqual(changes(record, old, self.policy["gates"]["validity"], self.policy), [])

    def test_unrelated_document_does_not_invalidate_arithmetic(self):
        record, old = self.run_validity()
        (self.source / "src/README.md").write_text("An unrelated document edit.\n")
        _, current, _ = self.snapshot()
        self.assertEqual(old, current)
        self.assertEqual(changes(record, current, self.policy["gates"]["validity"], self.policy), [])

    def test_package_change_reopens_every_identity_bound_gate(self):
        record, old = self.run_validity()
        self.package.write_text('["preserved package B"]\n')
        _, current, _ = self.snapshot()
        self.assertIn("input:package", changes(record, current, self.policy["gates"]["validity"], self.policy))
        self.assertTrue(self.status(old)["gates"]["validity"]["accepted"])
        self.assertFalse(self.status(current)["gates"]["validity"]["accepted"])

    def test_missing_and_wrong_snapshot_reviews_remain_open(self):
        _, manifest = self.run_validity()
        for snapshot in (None, "wrong snapshot"):
            self.review(manifest, snapshot=snapshot)
            self.assertFalse(self.status(manifest)["statuses"]["Compiler-closed"])

    def test_diagnostic_run_cannot_set_acceptance_status(self):
        _, manifest = self.run_validity(authority=False)
        self.review(manifest)
        result = self.status(manifest)
        self.assertFalse(result["statuses"]["Compiler-closed"])
        self.assertTrue(result["rejected"])

    def test_changed_result_and_log_are_rejected(self):
        _, manifest = self.run_validity()
        path = next((self.store / "runs").glob("*/result.json"))
        original = read_json(path)
        altered = copy.deepcopy(original)
        altered["record"]["outcome"] = "manual claim"
        write_json(path, altered)
        self.assertTrue(self.status(manifest)["rejected"])
        write_json(path, original)
        (path.parent / "command-0.log").write_text("substituted log")
        self.assertTrue(self.status(manifest)["rejected"])

    def test_changed_checker_or_gate_policy_cannot_keep_approval(self):
        for field in ("checker", "policy"):
            with self.subTest(field=field):
                self.authorize()
                approval = read_json(self.authority_path / "approval.json")
                approval[field] = "changed"
                write_json(self.authority_path / "approval.json", approval)
                with self.assertRaises(EvidenceError):
                    Authority(self.authority_path)

    def test_candidate_cannot_delete_a_required_gate(self):
        altered = copy.deepcopy(self.policy)
        altered["obligations"]["compiler"]["gates"] = []
        candidate = self.source / "candidate-policy.json"
        write_json(candidate, altered)
        with contextlib.redirect_stderr(io.StringIO()):
            status = main(["--source", str(self.source), "--store", str(self.store),
                           "--authority", str(self.authority_path), "--policy", str(candidate), "status"])
        self.assertEqual(status, 1)

    def test_wrappers_and_mentions_do_not_close_missing_target(self):
        self.policy["obligations"]["compiler"]["target"] = None
        self.authorize()
        _, manifest = self.run_validity()
        self.review(manifest)
        result = self.status(manifest)
        self.assertFalse(result["statuses"]["Compiler-closed"])
        self.assertIn("exact Lean target", markdown(result))

    def test_missing_required_case_is_failure_even_with_zero_exit(self):
        (self.source / "src/check.py").write_text('print("test result: ok. 0 passed; 0 failed;")\n')
        result, _ = self.run_validity()
        self.assertEqual(result["outcome"], "failed")
        self.assertIsNone(result["commands"][0]["completion"])

    def test_checked_source_mutation_is_not_a_pass(self):
        script = self.source / "src/check.py"
        script.write_text(script.read_text() + '\nopen("Meaning.lean", "w").write("changed")\n')
        result, _ = self.run_validity()
        self.assertEqual(result["outcome"], "failed")
        self.assertIn("checked input changed", result["reason"])

    def test_added_source_file_is_not_a_pass(self):
        script = self.source / "src/check.py"
        script.write_text(script.read_text() + '\nopen("Added.lean", "w").write("new input")\n')
        result, _ = self.run_validity()
        self.assertEqual(result["outcome"], "failed")

    def test_frozen_snapshot_mutation_is_rejected(self):
        _, manifest, directory = self.snapshot()
        (directory / "source/src/Meaning.lean").write_text("mutated")
        self.assertEqual((self.source / "src/Meaning.lean").read_text(), "def Meaning := 2\n")
        with self.assertRaises(EvidenceError):
            verify(directory, manifest)

    def test_candidate_cannot_replace_the_checker_library_seed(self):
        self.policy["inputs"]["library_seed"] = {}
        self.authorize()
        supplied = self.root / "supplied-inputs.json"
        write_json(supplied, {"library_seed": str(self.source / "src")})
        with contextlib.redirect_stderr(io.StringIO()) as error:
            status = main(["--source", str(self.source), "--store", str(self.store),
                           "--authority", str(self.authority_path), "--inputs", str(supplied), "status"])
        self.assertEqual(status, 1)
        self.assertIn("checker-owned library seed", error.getvalue())

    def test_library_seed_supplies_dependencies_for_a_clean_source_checkout(self):
        self.policy["sources"]["libraries"] = {"roots": ["formal/nightstream-fprime/.lake/packages"]}
        self.policy["inputs"]["library_seed"] = {}
        seed = self.root / "libraries"
        (seed / "mathlib/.git").mkdir(parents=True)
        (seed / "mathlib/Definition.lean").write_text("def Definition := True\n")
        (seed / "mathlib/.git/HEAD").write_text("preserved dependency Git metadata\n")
        self.inputs["library_seed"] = str(seed)
        _, manifest, _ = self.snapshot()
        self.assertIn("source/formal/nightstream-fprime/.lake/packages/mathlib/Definition.lean", entries(manifest))
        self.assertIn("inputs/library_seed/mathlib/.git/HEAD", entries(manifest))

    def test_snapshot_includes_untracked_files_and_omits_result_records(self):
        (self.source / "src/Untracked.lean").write_text("def Untracked := True\n")
        _, manifest, _ = self.snapshot()
        self.assertIn("source/src/Untracked.lean", entries(manifest))
        self.assertFalse(any("result.json" in path for path in entries(manifest)))

    def test_symlink_and_missing_external_input_are_not_passes(self):
        (self.source / "src/Alias.lean").symlink_to(self.source / "src/Meaning.lean")
        with self.assertRaises(EvidenceError):
            self.snapshot()
        (self.source / "src/Alias.lean").unlink()
        self.inputs.pop("fixture")
        _, manifest, directory = self.snapshot()
        with self.assertRaises(EvidenceError):
            run_gate("validity", self.policy, manifest, directory, self.store, self.authority)

    def test_lock_is_shared_between_stores(self):
        with build_lock(self.store):
            with self.assertRaises(EvidenceError):
                with build_lock(self.root / "other-store"):
                    self.fail("second managed gate acquired the lock")

    def test_relative_library_file_symlink_keeps_its_captured_target(self):
        alias = self.source / "src/Alias.lean"
        alias.symlink_to("Meaning.lean")
        _, manifest, directory = self.snapshot()
        self.assertTrue((directory / "source/src/Alias.lean").is_symlink())
        self.assertEqual((directory / "source/src/Alias.lean").read_text(), "def Meaning := 2\n")
        self.assertEqual(entries(manifest)["source/src/Alias.lean"]["link"], "Meaning.lean")

    def test_failed_retry_does_not_erase_matching_earlier_evidence(self):
        record, manifest = self.run_validity()
        failed = {**record, "outcome": "timed-out", "finished": "9999-12-31"}
        write_json(self.store / "runs/failed/result.json", self.authority.sign({**failed, "artifacts": {}}))
        self.assertTrue(self.status(manifest)["gates"]["validity"]["accepted"])

    def test_transitive_gate_dependencies_remain_required(self):
        self.policy["gates"]["parent"] = copy.deepcopy(self.policy["gates"]["validity"])
        self.policy["gates"]["parent"]["requires"] = ["validity"]
        self.policy["gates"]["grandparent"] = copy.deepcopy(self.policy["gates"]["parent"])
        self.policy["gates"]["grandparent"]["requires"] = ["parent"]
        self.authorize()
        _, manifest, directory = self.snapshot()
        for gate in ("parent", "grandparent"):
            run_gate(gate, self.policy, manifest, directory, self.store, self.authority)
        result = self.status(manifest)
        self.assertFalse(result["gates"]["grandparent"]["accepted"])

    def test_stage1_gap_does_not_relabel_closed_piccs_evidence(self):
        for name in ("compiler", "conformance"):
            self.policy["obligations"][name]["phase"] = "PiCCS"
        self.policy["obligations"]["production"]["phase"] = "Stage 1"
        self.authorize()
        _, manifest = self.run_validity()
        self.review(manifest)
        result = self.status(manifest)
        self.assertTrue(result["phase_statuses"]["PiCCS"]["Conformance-closed"])
        self.assertFalse(result["phase_statuses"]["Stage 1"]["Compiler-closed"])
        self.assertFalse(result["statuses"]["Conformance-closed"])

    def test_incomplete_metadata_cannot_be_a_passing_gate(self):
        command = self.policy["gates"]["validity"]["commands"][0]
        command.update(kind="lean", argv=["bash", "scripts/validate.sh", "file", "Fixture.lean"])
        self.authorize()
        _, manifest, directory = self.snapshot()
        def executed(_command, _work, log):
            Path(log).write_text("metadata emitted\n")
            return {"outcome": "pass", "completion": {"metadata": True}}
        with patch("scripts.lean_graph.runner.execute", side_effect=executed), \
             patch("scripts.lean_graph.runner.subprocess.run") as processes, \
             patch("scripts.lean_graph.metadata.from_log", return_value=[{
                 "root": "Target", "complete": False, "failures": ["unknown origin"]}]):
            processes.return_value.stdout = ""
            result = run_gate("validity", self.policy, manifest, directory, self.store, self.authority)
        self.assertEqual(result["outcome"], "incomplete")
        self.assertIn("command-0.log.gz", result["artifacts"])
        self.assertFalse(self.status(manifest)["gates"]["validity"]["accepted"])

    def test_caps_and_lean_rust_entrypoints_are_enforced(self):
        for kind, cap in CAPS.items():
            candidate = copy.deepcopy(self.policy)
            command = candidate["gates"]["validity"]["commands"][0]
            command.update(kind=kind, cap_seconds=cap + 1)
            with self.assertRaises(EvidenceError):
                validate(candidate)
        candidate = copy.deepcopy(self.policy)
        candidate["gates"]["validity"]["commands"][0]["kind"] = "lean"
        with self.assertRaises(EvidenceError):
            validate(candidate)

    def test_timeout_kills_process_group_and_records_failure(self):
        work = self.root / "work"
        (work / "source/src").mkdir(parents=True)
        command = {"kind": "python", "cwd": "src", "argv": [sys.executable, "-c", "pass"],
                   "completion": {"patterns": ["done"]}}
        # Simulate expiry at the project cap; do not wait five minutes for a harness test.
        with patch("scripts.lean_graph.runner.subprocess.Popen") as process, patch("os.killpg") as kill:
            child = process.return_value
            child.pid = os.getpid()
            child.communicate.side_effect = subprocess.TimeoutExpired(command["argv"], 300)
            child.returncode = -signal.SIGKILL
            result = execute(command, work, self.root / "timeout.log")
            self.assertEqual(result["outcome"], "timed-out")
            child.communicate.assert_called_once_with(None, timeout=300)
            kill.assert_called_with(child.pid, signal.SIGKILL)

    def test_interrupt_kills_process_group_and_records_failure(self):
        work = self.root / "work"
        (work / "source/src").mkdir(parents=True)
        command = {"kind": "python", "cwd": "src", "argv": [sys.executable, "-c", "pass"],
                   "completion": {"patterns": ["done"]}}
        with patch("scripts.lean_graph.runner.subprocess.Popen") as process, patch("os.killpg") as kill:
            child = process.return_value
            child.pid, child.returncode = os.getpid(), -signal.SIGKILL
            child.communicate.side_effect = InterruptedError("signal 15")
            result = execute(command, work, self.root / "interrupt.log")
            self.assertEqual(result["outcome"], "interrupted")
            kill.assert_called_with(child.pid, signal.SIGKILL)

    def test_incomplete_record_never_counts_as_pass(self):
        _, manifest = self.run_validity()
        path = next((self.store / "runs").glob("*/result.json"))
        record = self.authority.read(path)
        record["outcome"] = "incomplete"
        write_json(path, self.authority.sign(record))
        self.assertFalse(self.status(manifest)["gates"]["validity"]["accepted"])

    def test_cli_selects_active_criterion_and_reports_frozen_snapshot(self):
        identity, _, _ = self.snapshot()
        base = ["--source", str(self.source), "--store", str(self.store),
                "--authority", str(self.authority_path)]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(base + ["active", "compiler", "--evidence", "run validity"]), 0)
            self.assertEqual(main(base + ["--snapshot", identity, "run", "validity"]), 0)
            self.assertEqual(main(base + ["--snapshot", identity, "status"]), 0)
        self.assertIn("Compiler-closed", output.getvalue())
        self.assertIn(identity, output.getvalue())


if __name__ == "__main__":
    unittest.main()
