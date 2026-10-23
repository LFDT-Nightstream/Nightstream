import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


# These are the selections from the branch's CI before the runner change.
SELECTIONS = [shlex.split(group) for group in (
    "-p nightstream",
    "-p neo-spartan",
    "-p neo-ccs --test packed_witness",
    "-p neo-math --test signed_sums",
    "-p neo-ajtai --lib key::tests",
    "-p neo-ajtai --test nightstream_fprime_setup_parity --test nightstream_fprime_signed_commitment "
    "--test nightstream_fprime_prefix_commitment --test nightstream_fprime_matrix_commitment "
    "--test nightstream_fprime_batch_commitment",
    "-p neo-reductions --test matrix_rows",
    "-p neo-reductions --lib superneo_eval::openings::tests::",
    "-p nightstream-fprime --lib application_records",
    "-p nightstream-fprime --lib package::native_application",
    "-p neo-reductions --test pi_rlc_lean_parity",
    "-p nightstream-fprime --lib package::matrix_program",
    "-p nightstream-fprime --lib package::assignment_transport::quotient_tests",
    "-p nightstream-fprime --lib package::sealed::sealed_tests",
)]

FAKE_CARGO = r'''#!/usr/bin/env python3
import fcntl
import json
from pathlib import Path
import sys
import time

root = Path(__file__).parent
state = root / "state.json"
args = sys.argv[1:]
with (root / "state.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    data = json.loads(state.read_text())
    data["active"] += 1
    data["peak"] = max(data["peak"], data["active"])
    data["calls"].append(args)
    state.write_text(json.dumps(data))
code = 0
if args[:2] == ["nextest", "list"]:
    print(json.dumps({"build-arguments": args}))
elif "--binaries-metadata" in args:
    metadata = Path(args[args.index("--binaries-metadata") + 1])
    if not metadata.is_file() or "build-arguments" not in json.loads(metadata.read_text()):
        code = 2
time.sleep(0.1)
if (root / "fail").exists() and any(value.endswith("/9.json") for value in args):
    code = 101
with (root / "state.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    data = json.loads(state.read_text())
    data["active"] -= 1
    state.write_text(json.dumps(data))
raise SystemExit(code)
'''


def selection(arguments):
    selected = []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value in {"test", "nextest", "list", "run", "--release", "--locked",
                     "--no-run", "--no-fail-fast", "--doc"}:
            index += 1
        elif value in {"--config", "--list-type", "--message-format", "--profile",
                       "--binaries-metadata", "-E", "--filterset"}:
            index += 2
        elif value == "--":
            break
        else:
            selected.append(value)
            index += 1
    return selected


class CIRustTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        (self.root / "scripts").mkdir()
        source = Path(__file__).resolve().parents[1] / "ci_rust_tests.sh"
        self.script = self.root / "scripts/ci_rust_tests.sh"
        shutil.copyfile(source, self.script)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name in ("cargo", "cargo-nextest"):
            executable = self.bin / name
            executable.write_text(FAKE_CARGO)
            executable.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"])
        self.reset_calls()

    def reset_calls(self):
        (self.bin / "state.json").write_text(json.dumps({"active": 0, "peak": 0, "calls": []}))

    def run_mode(self, mode):
        result = subprocess.run(
            ["bash", str(self.script), mode], cwd=self.root, env=self.env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300,
        )
        self.calls = json.loads((self.bin / "state.json").read_text())
        self.assertEqual(self.calls["active"], 0, result.stdout)
        return result

    def assert_common_build_options(self, arguments):
        self.assertIn("--release", arguments)
        self.assertIn("--locked", arguments)
        configs = [arguments[index + 1] for index, value in enumerate(arguments) if value == "--config"]
        self.assertCountEqual(configs, ['profile.release.lto="off"', "profile.release.codegen-units=16"])
        self.assertNotIn("--features", arguments)
        self.assertNotIn("--all-features", arguments)

    def build_metadata(self):
        result = self.run_mode("build")
        self.assertEqual(result.returncode, 0, result.stdout)
        metadata = self.root / "target/ci-nextest"
        self.assertEqual({path.name for path in metadata.glob("*.json")},
                         {f"{index}.json" for index in range(14) if index != 11})
        self.assertEqual(len(self.calls["calls"]), 14)
        for index, expected in enumerate(SELECTIONS):
            if index == 11:
                continue
            arguments = json.loads((metadata / f"{index}.json").read_text())["build-arguments"]
            self.assertEqual(arguments[:2], ["nextest", "list"])
            self.assertEqual(arguments[arguments.index("--list-type") + 1], "binaries-only")
            self.assertEqual(arguments[arguments.index("--message-format") + 1], "json")
            self.assert_common_build_options(arguments)
            self.assertEqual(selection(arguments), expected[:-1] if "--lib" in expected else expected)
        native = [arguments for arguments in self.calls["calls"] if arguments[0] == "test"]
        self.assertEqual(len(native), 1)
        self.assertIn("--no-run", native[0])
        self.assert_common_build_options(native[0])
        self.assertEqual(selection(native[0]), SELECTIONS[11])
        self.reset_calls()

    def assert_runtime_coverage(self):
        calls = self.calls["calls"]
        self.assertEqual(len(calls), 15)
        self.assertEqual(self.calls["peak"], 4)
        native = [arguments for arguments in calls if arguments[0] == "test"]
        self.assertCountEqual([selection(arguments) for arguments in native],
                              [["-p", "nightstream", "--lib"], SELECTIONS[11]])
        for arguments in native:
            self.assert_common_build_options(arguments)
            self.assertIn("--no-fail-fast", arguments)
            self.assertNotIn("--no-run", arguments)
            self.assertEqual(arguments[-2:], ["--", "--test-threads=1"])
        seen = []
        for arguments in calls:
            if "--binaries-metadata" not in arguments:
                continue
            index = int(Path(arguments[arguments.index("--binaries-metadata") + 1]).stem)
            seen.append(index)
            self.assertIn("--no-fail-fast", arguments)
            self.assertEqual(arguments[arguments.index("--profile") + 1], "ci")
            self.assertNotIn("--no-run", arguments)
            expression = "".join(arguments[arguments.index("-E") + 1].split())
            expected = SELECTIONS[index]
            predicate = ("notkind(lib)" if index == 0 else
                         f"test(~{expected[-1]})" if "--lib" in expected else "all()")
            self.assertEqual(expression, predicate)
        self.assertCountEqual(seen, [index for index in range(14) if index != 11])

    def test_build_and_run_keep_all_original_selections(self):
        self.build_metadata()
        result = self.run_mode("test")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assert_runtime_coverage()

    def test_failure_fails_the_step_without_dropping_other_groups(self):
        self.build_metadata()
        (self.bin / "fail").touch()
        result = self.run_mode("test")
        self.assertNotEqual(result.returncode, 0)
        self.assert_runtime_coverage()

    def test_doctests_cover_the_two_original_default_selections(self):
        result = self.run_mode("docs")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(len(self.calls["calls"]), 2)
        self.assertCountEqual([selection(arguments) for arguments in self.calls["calls"]],
                              [["-p", "nightstream"], ["-p", "neo-spartan"]])
        for arguments in self.calls["calls"]:
            self.assertEqual(arguments[0], "test")
            self.assertIn("--doc", arguments)
            self.assertIn("--no-fail-fast", arguments)
            self.assertEqual(arguments[-2:], ["--", "--test-threads=1"])
            self.assert_common_build_options(arguments)


if __name__ == "__main__":
    unittest.main()
