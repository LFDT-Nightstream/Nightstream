import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import golden_conformance_ci as ci


class GoldenCITests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())

    def checked_cpu(self):
        self.write(self.root / "cpu/conformance.json", {"outcome": "passed"})
        context = [1, 2, 3, 4]
        caller = {"schema": 1, "verifier_context": context, "private_values": [7, 8],
                  "public_values": [9, 10], "output": [11], "output_digest": [12], "next_public_input": [13]}
        lean = [1, context, [7, 8], [9, 10], [[11], [12], [13], [14], [15], [16], [17]]]
        phase = [0] * 15
        phase[6], phase[14] = [14], [15]
        observed = {"pi_ccs_phase": phase, "outgoing_state": [16], "pi_rlc_parent": [0, [17]]}
        for step in (1, 2, 3):
            check = self.root / f"lean-step-{step}"
            self.write(check / "result.json", {"outcome": "passed"})
            values = {"proof.native": f"proof {step}".encode(), "pi_ccs_input.json": [step],
                      "children.json": [step], "actual_result.json": observed, "caller-inputs.json": caller}
            for name, value in values.items():
                self.write(self.root / f"cpu/fold-{step}" / name, value)
                self.write(check / f"inputs/fold-{step}" / name, value)
            for name in ("envelope.json", "fresh-claim.json"):
                self.write(self.root / f"cpu/step-{step}" / name, {"iteration": step})
                self.write(check / f"inputs/step-{step}" / name, {"iteration": step})
            self.write(check / f"step-{step}-lean-proof.native", values["proof.native"])
            self.write(self.root / f"cpu/fold-{step}/physical.bin", b"complete assignment")
            self.write(check / "lean-physical.bin", b"complete assignment")
            self.write(check / f"step-{step}-caller.json", lean)

    def test_handoff_compares_the_actual_lean_checked_bytes(self):
        self.checked_cpu()
        ci.cpu_handoff(self.root)
        changed = b"changed and rehashed CPU proof"
        self.write(self.root / "cpu/fold-3/proof.native", changed)
        self.write(self.root / "cpu-result.json", {
            "outcome": "passed", "sha256": hashlib.sha256(changed).hexdigest()})
        with self.assertRaisesRegex(ValueError, "CPU/Lean input handoff"):
            ci.cpu_handoff(self.root)
        self.write(self.root / "lean-step-3/inputs/fold-3/proof.native", changed)
        with self.assertRaisesRegex(ValueError, "CPU/fresh Lean proof handoff"):
            ci.cpu_handoff(self.root)

    def test_handoff_checks_caller_words_against_lean_after_snapshot_match(self):
        self.checked_cpu()
        path = self.root / "cpu/fold-2/caller-inputs.json"
        caller = json.loads(path.read_text())
        caller["private_values"][-1] += 1
        self.write(path, caller)
        self.write(self.root / "lean-step-2/inputs/fold-2/caller-inputs.json", caller)
        with self.assertRaisesRegex(ValueError, "every private caller word"):
            ci.cpu_handoff(self.root)

    def test_handoff_rejects_missing_lean_check_and_changed_physical_bytes(self):
        self.checked_cpu()
        self.write(self.root / "cpu/fold-1/physical.bin", b"changed assignment")
        with self.assertRaisesRegex(ValueError, "CPU/fresh Lean physical handoff"):
            ci.cpu_handoff(self.root)
        (self.root / "lean-step-1/result.json").unlink()
        with self.assertRaises(FileNotFoundError):
            ci.cpu_handoff(self.root)

    def test_cpu_requires_current_production_and_all_three_fresh_lean_checks(self):
        output = self.root / "new-cpu"
        with patch.object(ci, "run") as run, patch.object(ci, "build", return_value=Path("current-binary")) as build, \
                patch.object(ci, "cpu_handoff") as handoff:
            ci.execute("cpu", self.root / "archives", output)
        calls = [call.args[0] for call in run.call_args_list]
        native = [call for call in calls if Path(call[2]).name == "run_golden_conformance.py"]
        self.assertEqual(len(native), 1)
        self.assertEqual(native[0][-2:], ["--engine", "optimized"])
        lean = [call for call in calls if Path(call[2]).name == "check_lean_fold.py"]
        self.assertEqual([call[call.index("--step") + 1] for call in lean], [1, 2, 3])
        self.assertEqual(build.call_count, 2)
        handoff.assert_called_once_with(output)
        self.assertEqual(ci.load(output / "cpu-result.json")["outcome"], "passed")

    def test_failed_required_command_cannot_leave_a_success_receipt(self):
        output = self.root / "new-cpu"
        with patch.object(ci, "run", side_effect=ValueError("required check failed")), \
                self.assertRaisesRegex(ValueError, "required check failed"):
            ci.execute("cpu", self.root / "archives", output)
        self.assertFalse((output / "cpu-result.json").exists())

    def test_metal_requires_capability_and_same_cpu_handoff(self):
        with patch.object(ci.sys, "platform", "linux"), self.assertRaisesRegex(ValueError, "macOS runner"):
            ci.execute("metal", self.root / "archives", self.root / "metal", self.root / "handoff")
        with patch.object(ci.sys, "platform", "darwin"), self.assertRaisesRegex(ValueError, "CPU handoff"):
            ci.execute("metal", self.root / "archives", self.root / "metal")
        with patch.object(ci.sys, "platform", "darwin"), patch.object(ci, "run") as run, \
                patch.object(ci, "build", return_value=Path("current-metal")), patch.object(ci, "cpu_handoff") as handoff:
            ci.execute("metal", self.root / "archives", self.root / "metal", self.root / "handoff")
        self.assertEqual([call.args[0] for call in handoff.call_args_list], [self.root / "handoff"] * 2)
        self.assertEqual(run.call_args.args[0][-2:], ["--cpu-reference", self.root / "handoff/cpu"])

    def test_independent_mode_requires_real_generator_and_comparison_completion(self):
        output = self.root / "independent"
        with patch.object(ci, "run"), patch.object(ci, "build", return_value=Path("checker")), \
                patch.object(ci, "cpu_handoff"), \
                patch.object(ci, "independent_expectations", side_effect=ValueError("generation incomplete")) as replay, \
                self.assertRaisesRegex(ValueError, "generation incomplete"):
            ci.execute("independent", self.root / "archives", output, self.root / "handoff")
        replay.assert_called_once()
        self.assertFalse((output / "independent-result.json").exists())

    def test_build_rejects_success_without_the_requested_executable(self):
        def fake_run(command, **options):
            options["stdout"].write(json.dumps({"reason": "build-finished", "success": True}) + "\n")
            return type("Result", (), {"returncode": 0})()

        with patch.object(ci.subprocess, "run", side_effect=fake_run), \
                self.assertRaisesRegex(ValueError, "exactly one current executable"):
            ci.build(self.root, "optimized")


if __name__ == "__main__":
    unittest.main()
