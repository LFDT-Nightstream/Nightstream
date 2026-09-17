"""Runner contract tests. Subprocess execution is mocked; no producer runs."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

CANDIDATE = Path(__file__).parents[1] / "scripts/replay_recursive_loop.py"
SPEC = importlib.util.spec_from_file_location("recursive_loop", CANDIDATE)
loop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loop)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runner = loop.Replay.__new__(loop.Replay)
        self.runner.no_timeout = False
        self.enterContext(patch.dict(loop.os.environ, {"LEAN_SYSROOT": ""}))
        self.runner.root = self.root
        self.runner.directory = self.root / "step-2-to-3"
        self.runner.logs = self.runner.directory / "logs"
        self.runner.logs.mkdir(parents=True)
        self.enterContext(patch.object(loop, "REPO", self.root))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.execute = self.enterContext(patch.object(loop.subprocess, "run"))
        self.execute.return_value = SimpleNamespace(returncode=0)

    def checkpoint(self, name, argv, kind, outputs, exit_code):
        value = {"argv": list(map(str, argv)), "kind": kind, "cwd": str(self.root),
                 "exit": exit_code, "outputs": list(map(str, outputs)),
                 "output_identities": {str(path): loop.identity(path) for path in outputs if path.exists()}}
        (self.runner.logs / f"{name}.json").write_text(json.dumps(value))

    def test_same_cap_term_wrapper(self):
        for kind, cap, argv in [
                ("lean", "1500s", ["bash", "scripts/validate.sh", "lean-executable", "producer"]),
                ("python", "300s", ["python3", "check.py"]),
                ("rust", "300s", ["cargo", "run", "--release"]),
                ("static", "300s", ["cmp", "left", "right"])]:
            with self.subTest(kind=kind):
                self.runner.run("cap-" + kind, kind, self.root, argv)
                command = self.execute.call_args.args[0]
                position = command.index("timeout")
                self.assertEqual(command[position:position + 3], ["timeout", "--signal=TERM", cap])
                self.assertNotIn("--signal=KILL", command)
                self.assertEqual(command[command.index("--kind") + 1], kind)
                self.assertEqual(command[command.index("--") + 1:], argv)

    def test_owner_no_timeout_keeps_guard_without_deadline(self):
        self.runner.no_timeout = True
        self.runner.run("owner-no-timeout", "lean", self.root,
                        ["bash", "scripts/validate.sh", "lean-executable", "producer"])
        command = self.execute.call_args.args[0]
        self.assertNotIn("timeout", command)
        self.assertIn("--no-timeout", command)
        self.assertIn(str(self.root / "scripts/lean_graph/guard.py"), command)
        saved = loop.read(self.runner.logs / "owner-no-timeout.json")
        self.assertIsNone(saved["cap_seconds"])

    def test_no_timeout_reuses_matrix_sources_without_changing_ranges(self):
        self.runner.public = self.root / "public.json"
        self.runner.sources = self.root / "sources.jsonl"
        self.runner.rounds = self.root / "rounds"
        self.runner.package = self.root / "package.json"
        original_read = loop.read
        def input_record(path):
            if Path(path).name == "manifest.json":
                return {"outputs": [{"path": str(self.root / "parent-part.jsonl"),
                                     "start": 74272, "end": loop.BLOCKS}]}
            return original_read(path)
        for method, prefix, offset in [("ccs", "c-matrix-batch-", 5),
                                       ("reductions", "d-matrix-batch-", 4)]:
            with self.subTest(stage=method):
                variants = []
                for no_timeout in (False, True):
                    self.runner.no_timeout = no_timeout
                    with patch.object(self.runner, "lean") as lean, \
                            patch.object(self.runner, "python"), \
                            patch.object(self.runner, "rust"), \
                            patch.object(loop, "read", side_effect=input_record):
                        getattr(self.runner, method)()
                    variants.append([call for call in lean.call_args_list
                                     if call.args[0].startswith(prefix)])
                separate, shared = variants
                self.assertGreater(len(separate), 1)
                self.assertEqual(len(shared), 1)
                def requests(calls):
                    return [word for call in calls
                            for word in call.args[offset:call.args.index("--")]]
                self.assertEqual(requests(shared), requests(separate))
                self.assertEqual(shared[0].kwargs["outputs"],
                                 [path for call in separate for path in call.kwargs["outputs"]])

    def test_failed_checkpoint_does_not_resume(self):
        output = self.root / "partial.json"
        output.write_text("partial")
        argv = ["python3", "producer.py", str(output)]
        self.checkpoint("failed", argv, "python", [output], 124)
        with self.assertRaisesRegex(ValueError, "changed or failed checkpoint"):
            self.runner.run("failed", "python", self.root, argv, [output])
        self.execute.assert_not_called()
        self.assertEqual(output.read_text(), "partial")

    def test_missing_checkpoint_output_does_not_resume(self):
        output = self.root / "missing.json"
        argv = ["python3", "producer.py", str(output)]
        self.checkpoint("missing", argv, "python", [output], 0)
        with self.assertRaisesRegex(ValueError, "missing checkpoint output"):
            self.runner.run("missing", "python", self.root, argv, [output])
        self.execute.assert_not_called()

    def test_changed_output_contract_does_not_resume(self):
        first, second = self.root / "first.json", self.root / "second.json"
        first.write_text("one")
        second.write_text("two")
        argv = ["python3", "producer.py"]
        self.checkpoint("changed", argv, "python", [first], 0)
        with self.assertRaisesRegex(ValueError, "changed or failed checkpoint"):
            self.runner.run("changed", "python", self.root, argv, [first, second])
        self.execute.assert_not_called()

    def test_same_size_output_change_does_not_resume(self):
        output = self.root / "value.json"
        output.write_text("123")
        argv = ["python3", "producer.py", str(output)]
        self.checkpoint("value", argv, "python", [output], 0)
        output.write_text("124")
        with self.assertRaisesRegex(ValueError, "changed checkpoint output"):
            self.runner.run("value", "python", self.root, argv, [output])
        self.execute.assert_not_called()

    def test_missing_prefix_member_fails_startup_check(self):
        output = self.root / "prefix"
        output.mkdir()
        (output / "manifest.json").write_text("manifest")
        chunk = output / "part.bin"
        chunk.write_bytes(bytes([0, 1, 255]))
        self.checkpoint("prefix", ["producer"], "lean", [output], 0)
        chunk.unlink()
        with self.assertRaisesRegex(ValueError, "changed checkpoint output"):
            loop.check_saved_outputs(self.root)
        self.execute.assert_not_called()

    def test_copied_bootstrap_bytes_are_pinned(self):
        package = self.root / "original-package.json"
        package.write_text("pkg")
        original = self.root / "original-sources"
        original.mkdir()
        source = original / "witness.json"
        source.write_text("one")
        with patch.object(loop, "producer_sources", return_value={"source": "unchanged"}), \
                patch.object(loop.subprocess, "check_output", return_value="source-commit"):
            loop.pin_sources(self.root)
            for changed in (source, package):
                old = changed.read_text()
                changed.write_text("two")
                with self.assertRaisesRegex(ValueError, "producer sources changed"):
                    loop.pin_sources(self.root)
                changed.write_text(old)

    def test_selected_runtime_bytes_are_pinned(self):
        (self.root / "original-sources").mkdir()
        (self.root / "original-package.json").write_text("package")
        runtime = self.root / "runtime"
        files = [runtime / name for name in
                 ("bin/lean", "lib/lean/libleanshared.so", "lib/lean/libleanrt.a")]
        for path in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("original")
        with patch.object(loop, "producer_sources", side_effect=lambda: {"source": "unchanged"}), \
                patch.object(loop.subprocess, "check_output", return_value="source-commit"), \
                patch.dict(loop.os.environ, {"LEAN_SYSROOT": str(runtime)}):
            loop.pin_sources(self.root)
            for path in files:
                path.write_text("modified")
                with self.assertRaisesRegex(ValueError, "producer sources changed"):
                    loop.pin_sources(self.root)
                path.write_text("original")
            with patch.dict(loop.os.environ, {"LEAN_SYSROOT": ""}):
                with self.assertRaisesRegex(ValueError, "producer sources changed"):
                    loop.pin_sources(self.root)

    def test_extra_prefix_directory_changes_identity(self):
        output = self.root / "prefix"
        output.mkdir()
        (output / "manifest.json").write_text("manifest")
        self.checkpoint("prefix-directory", ["producer"], "lean", [output], 0)
        (output / "extra").mkdir()
        with self.assertRaisesRegex(ValueError, "changed checkpoint output"):
            loop.check_saved_outputs(self.root)

    def test_exact_handoff_pins_every_returned_child(self):
        self.runner.request = self.root / "request.json"
        prior = [2, [1, 2, 3, 4], [5, 6, 7, 8], [7, 11, 13, 17]]
        self.runner.request.write_text(json.dumps(prior))
        state = [9, 10, 11, 12]
        private = [0] * (49393 * 2 + 4)
        private[28] = 2
        private[30:34] = prior[1]
        private[35:39] = prior[2]
        private[49393 + 28] = 3
        private[49393 + 30:49393 + 34] = prior[1]
        private[49393 + 35:49393 + 39] = state
        private[-4:] = prior[3]
        caller = [1, list(map(int, loop.CONTEXT)), private, [], [state, 0, 0, 0, 0, 0, 0]]
        self.runner.out("caller.json").write_text(json.dumps(caller))
        native, material = self.runner.out("native-successor"), self.runner.out("native-material")
        native.mkdir()
        material.mkdir()
        (native / "envelope.json").write_text(json.dumps({"iteration": 3, "z0": prior[1], "current": state}))
        for child in range(16):
            (material / f"digit-{child}.json").write_text(str(child))
        self.runner.handoff()
        self.assertEqual(loop.read(self.runner.out("next-message-input.json")), [3, prior[1], state, prior[3]])
        for child in range(16):
            target = native / f"digit-{child}.json"
            self.assertFalse(target.is_symlink())
            self.assertTrue(target.samefile(material / target.name))
        loop.check_saved_outputs(self.root)
        (native / "digit-15.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed checkpoint output"):
            loop.check_saved_outputs(self.root)
        self.execute.assert_not_called()

    def test_full_loop_completes_both_successors_before_terminal(self):
        events = []
        class RecordedReplay:
            def __init__(self, root, iteration, no_timeout=False):
                self.iteration = iteration
            def __getattr__(self, name):
                return lambda: events.append((self.iteration, name))
        with patch.object(loop, "Replay", RecordedReplay), \
                patch("sys.argv", [str(CANDIDATE), str(self.root), "2", "all"]):
            loop.main()
        self.assertEqual(events, [(2, "build")] + [(iteration, phase) for iteration in (2, 3)
            for phase in ("prepare", "native", "ccs", "reductions", "successor")] + [(3, "terminal")])

    def test_failed_invocation_records_failure(self):
        self.execute.return_value = SimpleNamespace(returncode=7)
        with self.assertRaisesRegex(RuntimeError, "failed: inspect"):
            self.runner.run("new-failure", "python", self.root, ["python3", "producer.py"])
        saved = json.loads((self.runner.logs / "new-failure.json").read_text())
        self.assertEqual(saved["exit"], 7)

    def test_zero_exit_without_output_fails(self):
        output = self.root / "missing.json"
        with self.assertRaisesRegex(RuntimeError, "did not produce its required outputs"):
            self.runner.run("missing-new", "python", self.root,
                            ["python3", "producer.py"], [output])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
