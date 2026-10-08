import contextlib
import copy
import errno
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
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
                     patch.object(check.time, "monotonic", side_effect=[0, 0, cap, cap]), \
                     patch.object(check, "kill_process_groups") as kill:
                    with self.assertRaisesRegex(ValueError, "failed; see"):
                        checker.phase("capped", kind, ["test"])
                kill.assert_called_once_with(123)
                self.assertLessEqual(process.wait.call_args_list[0].kwargs["timeout"], cap)
                self.assertTrue(popen.call_args.kwargs["start_new_session"])
                if kind == "lean":
                    self.assertEqual(popen.call_args.kwargs["env"]["LEAN_TIMEOUT_SECONDS"], str(cap))
                record = json.loads((output / "capped.command.json").read_text())
                self.assertEqual((record["exit"], record["outcome"], record["cap_seconds"]), (124, "timed-out", cap))
                self.assertTrue((output / "capped.log").exists())

    def test_suspension_counts_toward_cap_even_when_child_exits_successfully(self):
        for child_exits in (False, True):
            with self.subTest(child_exits=child_exits):
                output = self.directory / str(child_exits)
                output.mkdir()
                checker = check.Check(self.directory, 1, output, Path("unused-native-checker"))
                process = MagicMock(pid=123)
                process.wait.side_effect = ([0, 0] if child_exits else
                                           [subprocess.TimeoutExpired(["test"], 1), -9])
                clock = [0, 0, 301] if child_exits else [0, 0, 301, 301]
                with patch.object(check, "build_lock", return_value=contextlib.nullcontext()), \
                     patch.object(check, "check_build_processes"), \
                     patch.object(check.subprocess, "Popen", return_value=process), \
                     patch.object(check.time, "monotonic", return_value=0), \
                     patch.object(check.time, "time", side_effect=clock), \
                     patch.object(check, "kill_process_groups") as kill:
                    with self.assertRaisesRegex(ValueError, "failed; see"):
                        checker.phase("suspended", "python", ["test"])
                kill.assert_called_once_with(123)
                self.assertLessEqual(process.wait.call_args_list[0].kwargs["timeout"], 1)
                record = json.loads((output / "suspended.command.json").read_text())
                self.assertEqual((record["exit"], record["outcome"]), (124, "timed-out"))
                self.assertEqual(record["elapsed_seconds"], 301)

    def test_successful_child_data_and_command_receipt_have_distinct_paths(self):
        checker = check.Check(self.directory, 1, self.directory, Path("unused-native-checker"))
        data = self.directory / "step-1-caller.json"
        command = [sys.executable, "-c",
                   "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('{\"schema\":1}'); print('emitted caller')",
                   str(data)]
        with patch.object(check, "build_lock", return_value=contextlib.nullcontext()), \
             patch.object(check, "check_build_processes"):
            self.assertEqual(checker.phase("step-1-caller", "python", command), "emitted caller\n")
        self.assertEqual(data.read_text(), '{"schema":1}')
        receipt = json.loads((self.directory / "step-1-caller.command.json").read_text())
        self.assertEqual((receipt["outcome"], receipt["exit"]), ("passed", 0))
        self.assertEqual(receipt["command"], command)

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

    def test_blocked_handoff_does_not_replace_actual_c_rejection(self):
        manifest = {"cases": [
            {"case": "public_child_public", "file": "mutations/public_child_public.json", "expected_owner": "public_check"},
            {"case": "encoding_short_point", "file": "mutations/encoding_short_point.json", "expected_owner": "decoder"},
            {"case": "invalid_first_round_constant", "file": "invalid_first_round_constant.json", "expected_owner": "upstream_pi_ccs"},
        ]}
        log = "\n".join((
            "lean_pi_dec_mutation=public_child_public.json rejected_by=public_check",
            "lean_pi_dec_mutation=encoding_short_point.json rejected_by=decoder reason=short point",
            "lean_pi_dec_mutations=passed public=1 encoding=1 unbounded=1 rejected_C_stops_D=1",
        ))
        with self.assertRaisesRegex(ValueError, "missing Lean rejection: invalid_first_round_constant"):
            check.mutation_rejections(manifest, log)
        complete = "lean_pi_ccs_mutation=invalid_first_round_constant rejected_by=pi_ccs\n" + log
        owners = check.mutation_rejections(manifest, complete)
        self.assertEqual(owners["pi_ccs"], ["invalid_first_round_constant"])
        self.assertEqual(owners["internal"], ["unbounded_parent", "rejected_C_stops_D"])
        with self.assertRaisesRegex(ValueError, "incomplete Lean mutation result"):
            check.mutation_rejections(manifest, complete.replace("rejected_C_stops_D=1", "rejected_C_stops_D=0"))


class WaitForExitTests(unittest.TestCase):
    def child(self, code):
        process = subprocess.Popen([sys.executable, "-c", code])
        self.addCleanup(process.wait)
        self.addCleanup(process.kill)
        return process

    def test_returns_the_exit_code_before_the_timeout(self):
        process = self.child("import sys; sys.exit(3)")
        started = time.monotonic()
        self.assertEqual(check.wait_for_exit(process, 60), 3)
        self.assertLess(time.monotonic() - started, 30)

    def test_running_process_times_out_like_wait(self):
        process = self.child("import time; time.sleep(60)")
        with self.assertRaises(subprocess.TimeoutExpired):
            check.wait_for_exit(process, 0.2)
        self.assertIsNone(process.poll())

    def test_exit_notification_uses_remaining_time_to_reap(self):
        process = MagicMock(spec=check.POPEN, pid=123, returncode=None)

        def reap(*, timeout):
            if timeout == 0:
                raise subprocess.TimeoutExpired(["child"], timeout)
            return 3

        process.wait.side_effect = reap
        selector = MagicMock(KQ_EV_ERROR=0x4000)
        selector.kqueue.return_value.control.return_value = [MagicMock(flags=0)]
        with patch.object(check, "select", selector), \
             patch.object(check.time, "monotonic", side_effect=[0, 0.25]):
            self.assertEqual(check.wait_for_exit(process, 1), 3)
        process.wait.assert_called_once_with(timeout=0.75)

    def test_notification_timeout_does_not_restart_deadline(self):
        for elapsed in (1, 1.25):
            with self.subTest(elapsed=elapsed):
                process = MagicMock(spec=check.POPEN, pid=123, returncode=None)
                process.wait.side_effect = subprocess.TimeoutExpired(["child"], 0)
                selector = MagicMock(KQ_EV_ERROR=0x4000)
                selector.kqueue.return_value.control.return_value = []
                with patch.object(check, "select", selector), \
                     patch.object(check.time, "monotonic", side_effect=[0, elapsed]):
                    with self.assertRaises(subprocess.TimeoutExpired):
                        check.wait_for_exit(process, 1)
                process.wait.assert_called_once_with(timeout=0)

    def test_kqueue_registration_error_uses_timed_wait(self):
        process = self.child("import time; time.sleep(60)")
        selector = MagicMock(KQ_EV_ERROR=0x4000)
        queue = selector.kqueue.return_value
        queue.control.return_value = [MagicMock(flags=selector.KQ_EV_ERROR, data=errno.ENOMEM)]
        with patch.object(check, "select", selector), \
             patch.object(check.time, "monotonic", return_value=0):
            with self.assertRaises(subprocess.TimeoutExpired) as expired:
                check.wait_for_exit(process, 0.2)
        self.assertEqual(expired.exception.timeout, 0.2)
        self.assertIsNone(process.poll())
        queue.close.assert_called_once_with()

    def test_pidfd_select_rejection_uses_timed_wait_and_closes_descriptor(self):
        process = self.child("import time; time.sleep(60)")
        selector = MagicMock(spec=["select"])
        selector.select.side_effect = ValueError("filedescriptor out of range in select()")
        with patch.object(check, "select", selector), \
             patch.object(check.time, "monotonic", return_value=0), \
             patch.object(check.os, "pidfd_open", return_value=1024, create=True), \
             patch.object(check.os, "close") as close:
            with self.assertRaises(subprocess.TimeoutExpired) as expired:
                check.wait_for_exit(process, 0.2)
        self.assertEqual(expired.exception.timeout, 0.2)
        self.assertIsNone(process.poll())
        close.assert_called_once_with(1024)

    @unittest.skipUnless(hasattr(os, "waitid"), "os.waitid requires Python 3.13+ on macOS")
    def test_exited_unreaped_process_is_reaped(self):
        process = self.child("pass")
        # Wait for the exit without reaping, so that the process is a zombie.
        os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
        self.assertEqual(check.wait_for_exit(process, 60), 0)
        self.assertEqual(process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
