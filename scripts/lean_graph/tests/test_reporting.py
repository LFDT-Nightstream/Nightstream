import contextlib
import io
import os
import subprocess
import sys
from unittest.mock import patch

from scripts.lean_graph import guard
from scripts.lean_graph.evidence import main
from scripts.lean_graph.records import explain, markdown
from scripts.lean_graph.snapshot import read_json, write_json
from scripts.lean_graph.tests.test_evidence import EvidenceFixture
from scripts.lean_graph.tests import test_workflow as workflow


class ReportingTests(EvidenceFixture):
    def test_stale_report_names_changed_added_and_removed_files(self):
        (self.source / "src/Removed.lean").write_text("def Removed := True\n")
        self.run_validity()
        (self.source / "src/Meaning.lean").write_text("def Meaning := 3\n")
        (self.source / "src/Added.lean").write_text("def Added := True\n")
        (self.source / "src/Removed.lean").unlink()
        _, current, _ = self.snapshot()
        report = self.status(current)
        files = report["stale"][0]["details"]["files"]
        self.assertEqual({item["path"]: item["change"] for item in files}, {
            "source/src/Added.lean": "added", "source/src/Meaning.lean": "changed",
            "source/src/Removed.lean": "removed"})
        self.assertIn("source/src/Meaning.lean", markdown(report))
        self.assertFalse(report["gates"]["validity"]["accepted"])

    def test_gate_timing_includes_work_outside_commands(self):
        record, manifest = self.run_validity()
        times = record["timings_seconds"]
        self.assertGreater(record["elapsed_seconds"], 0)
        self.assertAlmostEqual(sum(times.values()), record["elapsed_seconds"])
        self.assertGreater(times["commands"], 0)
        self.assertGreater(times["preparation"], 0)
        self.assertGreater(times["input_validation"], 0)
        self.assertIn("Gate total:", explain(self.status(manifest), "compiler"))

    def test_explicit_checkpoint_does_not_overwrite_another_tasks_selection(self):
        selected = {"obligation": "production", "evidence": "another task"}
        write_json(self.store / "active.json", selected)
        policy, inputs = self.root / "policy.json", self.root / "inputs.json"
        write_json(policy, self.policy)
        write_json(inputs, self.inputs)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main(["--source", str(self.source), "--store", str(self.store), "--policy", str(policy),
                         "--inputs", str(inputs), "checkpoint", "compiler"])
        self.assertEqual(code, 0)
        self.assertEqual(read_json(self.store / "active.json"), selected)

    def test_guard_shares_lock_across_clients_with_different_tmpdir(self):
        code = ("from pathlib import Path; from scripts.lean_graph import guard; import sys; "
                "guard._LOCK_PATH=Path(sys.argv[1]); "
                "sys.exit(guard.main(['--kind','python','--',sys.executable,'-c','print(42)']))")
        with guard.build_lock():
            result = subprocess.run([sys.executable, "-B", "-c", code, str(guard._LOCK_PATH)],
                env={**os.environ, "TMPDIR": str(self.root)}, capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 1)
        self.assertIn("holds the shared build lock", result.stderr)
        self.assertNotIn("42", result.stdout)

    def test_guard_applies_project_cap_and_releases_after_failure(self):
        with patch.object(guard, "check_build_processes"), patch.dict(guard.CAPS, {"python": 0.1}):
            result = guard.run([sys.executable, "-c", "import time; time.sleep(300)"], "python", self.root)
        self.assertEqual(result["outcome"], "timed-out")
        self.assertEqual(result["exit"], 124)
        self.assertEqual(result["cap_seconds"], 0.1)
        with guard.build_lock():
            pass

    def test_guard_rejects_debug_rust_and_direct_lean(self):
        for command, kind in ((["cargo", "test"], "rust"), (["lake", "build"], "lean")):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(guard.main(["--kind", kind, "--", *command]), 1)


class DeclarationReportingTests(EvidenceFixture):
    prepare_freshness = workflow.WorkflowTests.prepare_freshness
    scoped_snapshot = workflow.WorkflowTests.scoped_snapshot
    export = workflow.WorkflowTests.export

    def test_stale_report_names_changed_registered_proof_roots(self):
        self.prepare_freshness()
        _, old, directory = self.scoped_snapshot(["validity"])
        self.export(old, directory)
        self.run_validity()
        (self.source / "src/Meaning.lean").write_text("def Meaning := 4\n")
        _, current, directory = self.scoped_snapshot(["validity"])
        self.export(current, directory, proof="changed-proof")
        stale = next(item for item in self.status(current)["stale"] if item["gate"] == "validity")
        self.assertEqual(stale["details"]["declarations"], [
            {"name": "Test.closure", "key": "proof", "change": "changed"}])
