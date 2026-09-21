import contextlib
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch


SPEC = importlib.util.spec_from_file_location("check_lean_fold", Path(__file__).with_name("check_lean_fold.py"))
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


class LeanFoldCheckTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def caller(self):
        context = [1, 2, 3, 4]
        native = {"schema": 1, "verifier_context": context, "private_values": [7, 8],
                  "public_values": [9, 10], "output": [11], "output_digest": [12], "next_public_input": [13]}
        lean = [1, context, [7, 8], [9, 10], [[11], [12], [13], [14], [15], [16], [17]]]
        phase = [0] * 15
        phase[6], phase[14] = [14], [15]
        observed = {"pi_ccs_phase": phase, "outgoing_state": [16], "pi_rlc_parent": [0, [17]]}
        return native, lean, observed, context

    def test_complete_caller_comparison_rejects_tail_words_and_every_result_family(self):
        native, lean, observed, context = self.caller()
        self.assertTrue(check.compare_caller(native, lean, observed, context)["complete_caller_word_equality_checked"])
        for field in ("private_values", "public_values", "verifier_context"):
            with self.subTest(field=field):
                changed = copy.deepcopy(native)
                changed[field][-1] += 1
                with self.assertRaises(ValueError):
                    check.compare_caller(changed, lean, observed, context)
        for index in range(len(lean[4])):
            with self.subTest(result=index):
                changed = copy.deepcopy(lean)
                changed[4][index][-1] += 1
                with self.assertRaises(ValueError):
                    check.compare_caller(native, changed, observed, context)

    def test_carried_identity_cannot_select_the_key(self):
        pin = self.directory / "identity.rs"
        pin.write_text("// pub const POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY: [u64; 4] = [9,9,9,9];\n"
                       "pub const POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY: [u64; 4] = [1,2,3,4];\n")
        selected = check.package_pin(pin)
        self.assertEqual(selected, [1, 2, 3, 4])
        (self.directory / "actual_result.json").write_text(json.dumps({"package_identity": [9, 9, 9, 9]}))
        checker = check.Check(self.directory, 1, self.directory, Path("unused-native-checker"))
        with patch.object(checker, "phase") as phase:
            with self.assertRaisesRegex(ValueError, "selected package identity"):
                checker.verify_fold(1, self.directory, selected, Path("package.json"))
            phase.assert_not_called()

    def test_timeout_kills_process_group_and_preserves_failure_receipt(self):
        for kind, cap in (("lean", 1500), ("native", 300), ("python", 300)):
            with self.subTest(kind=kind):
                output = self.directory / kind
                output.mkdir()
                checker = check.Check(self.directory, 1, output, Path("unused-native-checker"))
                process = MagicMock(pid=123)
                process.wait.side_effect = [subprocess.TimeoutExpired(["test"], cap), -9]
                with patch.object(check, "build_lock", return_value=contextlib.nullcontext()), \
                     patch.object(check, "check_build_processes"), \
                     patch.object(check.subprocess, "Popen", return_value=process) as popen, \
                     patch.object(check.os, "killpg") as kill:
                    with self.assertRaisesRegex(ValueError, "failed; see"):
                        checker.phase("capped", kind, ["test"])
                kill.assert_called_once_with(123, check.signal.SIGKILL)
                self.assertLessEqual(process.wait.call_args_list[0].kwargs["timeout"], cap)
                self.assertTrue(popen.call_args.kwargs["start_new_session"])
                if kind == "lean":
                    self.assertEqual(popen.call_args.kwargs["env"]["LEAN_TIMEOUT_SECONDS"], str(cap))
                record = json.loads((output / "capped.json").read_text())
                self.assertEqual((record["exit"], record["outcome"], record["cap_seconds"]), (124, "timed-out", cap))
                self.assertTrue((output / "capped.log").exists())

    def test_source_snapshots_detect_later_changed_content(self):
        original = self.directory / "original.json"
        original.write_bytes(b"original")
        output = self.directory / "output"
        output.mkdir()
        checker = check.Check(self.directory, 1, output, Path("unused-native-checker"))
        snapshot = checker.snapshot(original, "original.json")
        self.assertEqual(snapshot.read_bytes(), original.read_bytes())
        original.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "complete bytes differ"):
            check.compare_files(original, snapshot, "original")


if __name__ == "__main__":
    unittest.main()
