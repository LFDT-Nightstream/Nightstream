import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "run_golden_conformance", Path(__file__).with_name("run_golden_conformance.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class GoldenConformanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.binary = self.root / "native-test"
        self.binary.write_bytes(b"test binary identity")
        self.directory, self.references = self.root / "run", self.root / "references"
        for name in ("reference-first", "reference-later"):
            (self.references / name).mkdir(parents=True)
        self.calls = []
        self.missing_phase = self.failed_phase = self.wrong_exit_phase = None

    @staticmethod
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def execute(self, command, *, cwd, check):
        self.calls.append(command)
        self.assertTrue(check)
        self.assertEqual(command[:3], ["timeout", "--signal=KILL", "300"])
        options = dict(zip(command[6::2], command[7::2]))
        directory = Path(options["--directory"])
        if Path(command[5]).name == "run_recursive_phase.py":
            name = options["--phase"]
            if name == self.failed_phase:
                raise subprocess.CalledProcessError(1, command)
            if name == self.missing_phase:
                return subprocess.CompletedProcess(command, 0)
            request = {"phase": name, "directory": str(directory)}
            for key, value in options.items():
                if key not in ("--binary", "--directory", "--phase"):
                    request[key[2:].replace("-", "_")] = int(value) if key in ("--step", "--child") else value
            parts = [name] + [f"{key}-{request[key]}" for key in ("step", "child") if key in request]
            self.save(directory / "logs" / ("-".join(parts) + ".json"), {
                "request": request, "exit": 0, "outcome": "passed", "cap_seconds": 300,
                "process_exit": 1 if name == self.wrong_exit_phase else 0,
            })
        else:
            self.assertEqual(Path(command[5]).name, "compare_recursive_outputs.py")
            engine = options.get("--engine-directory")
            self.save(Path(options["--receipt"]), {
                "outcome": "passed", "fold": int(options["--fold"]),
                "run_directory": str(directory), "reference_directory": options.get("--reference"),
                "later_fold": {"engine_comparisons": [{"engine_directory": engine}] if engine else []},
            })
        return subprocess.CompletedProcess(command, 0)

    def invoke(self, engine="optimized", cpu=None):
        argv = ["run_golden_conformance.py", "--binary", str(self.binary),
                "--directory", str(self.directory), "--references", str(self.references), "--engine", engine]
        if cpu is not None:
            argv += ["--cpu-reference", str(cpu)]
        with patch.object(sys, "argv", argv), patch.object(runner, "source_identity", return_value={"commit": "test"}), \
                patch.object(runner.subprocess, "run", side_effect=self.execute), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return runner.main()

    def phase_calls(self):
        return [dict(zip(command[6::2], command[7::2])) for command in self.calls
                if Path(command[5]).name == "run_recursive_phase.py"]

    def test_cpu_runs_selected_folds_with_one_openings_phase_each(self):
        self.assertEqual(self.invoke(), 0)
        phases = self.phase_calls()
        expected = ["base"]
        for step in (1, 2):
            expected += ["sources", "ccs", "rlc", "split", "openings", "nifs", "successor"]
        expected += ["terminal", "mutation", "reject"]
        self.assertEqual([call["--phase"] for call in phases], expected)
        self.assertEqual([call["--step"] for call in phases if call["--phase"] == "openings"],
                         ["1", "2"])
        self.assertEqual([call["--step"] for call in phases[-3:]], ["3", "3", "3"])
        self.assertEqual([dict(zip(call[6::2], call[7::2]))["--fold"] for call in self.calls[-2:]], ["1", "2"])
        self.assertEqual(runner.read(self.directory / "conformance.json")["outcome"], "passed")

    def test_metal_uses_cpu_proofs_and_preserves_previous_cpu_receipt(self):
        cpu = self.root / "cpu"
        for step in (1, 2):
            path = cpu / f"fold-{step}/proof.native"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"CPU canonical proof")
        previous = cpu / "comparison-fold-2.json"
        previous.write_bytes(b"previous successful CPU comparison")
        self.assertEqual(self.invoke("metal", cpu), 0)
        phases = self.phase_calls()
        self.assertEqual([call["--phase"] for call in phases], [
            "base", "sources", "prove", "successor", "sources", "prove", "successor",
            "terminal", "mutation", "reject"])
        self.assertEqual([call["--reference-proof"] for call in phases if call["--phase"] == "prove"],
                         [str(cpu / f"fold-{step}/proof.native") for step in (1, 2)])
        comparison = dict(zip(self.calls[-1][6::2], self.calls[-1][7::2]))
        self.assertEqual(comparison["--directory"], str(cpu))
        self.assertEqual(comparison["--engine-directory"], str(self.directory))
        self.assertEqual(comparison["--fold"], "2")
        self.assertEqual(comparison["--reference"], str(self.references / "reference-later"))
        self.assertEqual(previous.read_bytes(), b"previous successful CPU comparison")
        self.assertTrue((self.directory / "comparison-lean-cpu-engine.json").is_file())
        self.assertFalse((self.directory / "comparison-cpu-engine-fold-3.json").exists())

    def test_missing_phase_receipt_cannot_count_as_success(self):
        self.missing_phase = "openings"
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.phase_calls()[-1]["--phase"], "openings")
        self.assertEqual(runner.read(self.directory / "conformance.json")["outcome"], "failed")

    def test_failed_subprocess_stops_before_next_phase(self):
        self.failed_phase = "split"
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.phase_calls()[-1]["--phase"], "split")
        self.assertFalse((self.directory / "comparison-fold-2.json").exists())

    def test_passed_receipt_cannot_hide_failed_test_exit(self):
        self.wrong_exit_phase = "base"
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(len(self.calls), 1)

    def test_old_run_cannot_replace_fresh_execution(self):
        self.directory.mkdir()
        self.save(self.directory / "conformance.json", {"outcome": "passed"})
        with self.assertRaisesRegex(ValueError, "run directory already exists"):
            self.invoke()
        self.assertEqual(self.calls, [])
        self.assertEqual(runner.read(self.directory / "conformance.json"), {"outcome": "passed"})


if __name__ == "__main__":
    unittest.main()
