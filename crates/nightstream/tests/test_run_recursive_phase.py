import importlib.util
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryFile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location(
    "run_recursive_phase", Path(__file__).with_name("run_recursive_phase.py"))
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def usage(peak=0):
    return SimpleNamespace(ru_utime=0, ru_stime=0, ru_maxrss=peak)


class RecursivePhaseMemoryTests(unittest.TestCase):
    def completed_test(self, peak, platform):
        with TemporaryFile() as output, patch.object(RUNNER.sys, "platform", platform), \
                patch.object(RUNNER.resource, "getrusage", side_effect=[usage(), usage(peak)]):
            return RUNNER.run_test(
                [sys.executable, "-c", "import sys; sys.stdin.read(); print('completed')"], {}, output)

    def test_fast_exit_peak_over_guard_fails_on_both_platforms(self):
        for platform, unit in (("linux", 1024), ("darwin", 1)):
            with self.subTest(platform=platform):
                result = self.completed_test(RUNNER.RSS_CAP_BYTES // unit + 1, platform)
                self.assertEqual(result["process_exit"], 0)
                self.assertEqual(result["outcome"], "memory-cap")
                self.assertNotEqual(result["exit"], 0)
                self.assertTrue(result["memory_cap_exceeded"])
                self.assertIn("after exit", result["memory_enforcement"])

    def test_exact_guard_is_accepted(self):
        for platform, unit in (("linux", 1024), ("darwin", 1)):
            with self.subTest(platform=platform):
                result = self.completed_test(RUNNER.RSS_CAP_BYTES // unit, platform)
                self.assertEqual(result["outcome"], "passed")
                self.assertFalse(result["memory_cap_exceeded"])
                self.assertEqual(result["maximum_resident_bytes"], RUNNER.RSS_CAP_BYTES)

    def pending_test(self, observed):
        process = Mock()
        process.pid = 123
        process.returncode = -9
        process.poll.return_value = None
        process.communicate.side_effect = [subprocess.TimeoutExpired("test", 1), (None, None)]
        with TemporaryFile() as output, patch.object(RUNNER.sys, "platform", "linux"), \
                patch.object(RUNNER.subprocess, "Popen", return_value=process), \
                patch.object(RUNNER.resource, "getrusage", side_effect=[usage(), usage()]), \
                patch.object(RUNNER, "resident_bytes", side_effect=observed):
            result = RUNNER.run_test(["test"], {}, output)
        process.kill.assert_called_once()
        return result

    def test_observed_excess_stops_the_child(self):
        result = self.pending_test([RUNNER.RSS_CAP_BYTES + 1])
        self.assertEqual(result["outcome"], "memory-cap")
        self.assertEqual(result["memory_enforcement"], "killed on observed RSS excess")
        self.assertEqual(result["maximum_resident_bytes"], RUNNER.RSS_CAP_BYTES + 1)

    def test_monitoring_sends_the_request_only_once(self):
        process = Mock()
        process.returncode = 0
        process.poll.return_value = 0
        process.communicate.side_effect = [subprocess.TimeoutExpired("test", 1),
                                           (None, None), (None, None)]
        with TemporaryFile() as output, patch.object(RUNNER.sys, "platform", "linux"), \
                patch.object(RUNNER.subprocess, "Popen", return_value=process), \
                patch.object(RUNNER.resource, "getrusage", side_effect=[usage(), usage()]), \
                patch.object(RUNNER, "resident_bytes", return_value=1):
            result = RUNNER.run_test(["test"], {"phase": "base"}, output)
        self.assertEqual(result["outcome"], "passed")
        calls = process.communicate.call_args_list
        self.assertEqual(calls[0].args, (b'{"phase": "base"}',))
        self.assertEqual(calls[1].args, (None,))
        process.kill.assert_not_called()

    def test_live_observation_failure_cannot_pass(self):
        result = self.pending_test([OSError("observer failed")])
        self.assertEqual(result["outcome"], "memory-observation-failed")
        self.assertNotEqual(result["exit"], 0)

    def test_missing_rss_during_exit_waits_for_status_and_checks_final_peak(self):
        for peak, outcome in ((0, "passed"), (RUNNER.RSS_CAP_BYTES // 1024 + 1, "memory-cap")):
            with self.subTest(outcome=outcome):
                process = Mock()
                process.pid = 123
                process.returncode = None
                process.poll.return_value = None

                def communicate(*args, **kwargs):
                    if process.communicate.call_count == 1:
                        raise subprocess.TimeoutExpired("test", RUNNER.RSS_POLL_SECONDS)
                    process.returncode = 0
                    process.poll.return_value = 0
                    return None, None

                process.communicate.side_effect = communicate
                with TemporaryFile() as output, patch.object(RUNNER.sys, "platform", "linux"), \
                        patch.object(RUNNER.subprocess, "Popen", return_value=process), \
                        patch.object(RUNNER.resource, "getrusage", side_effect=[usage(), usage(peak)]), \
                        patch.object(RUNNER, "resident_bytes", return_value=None):
                    result = RUNNER.run_test(["test"], {}, output)
                self.assertEqual(result["outcome"], outcome)
                self.assertEqual(result["process_exit"], 0)
                self.assertIsNone(result["memory_observation_error"])
                self.assertLessEqual(process.communicate.call_args_list[1].kwargs["timeout"], RUNNER.CAPS["rust"])
                process.kill.assert_not_called()

    def test_missing_rss_without_exit_still_times_out(self):
        process = Mock()
        process.returncode = -9
        process.poll.return_value = None
        process.communicate.side_effect = [subprocess.TimeoutExpired("test", RUNNER.RSS_POLL_SECONDS),
                                           subprocess.TimeoutExpired("test", RUNNER.CAPS["rust"]),
                                           (None, None)]
        with TemporaryFile() as output, patch.object(RUNNER.sys, "platform", "linux"), \
                patch.object(RUNNER.subprocess, "Popen", return_value=process), \
                patch.object(RUNNER.resource, "getrusage", side_effect=[usage(), usage()]), \
                patch.object(RUNNER, "resident_bytes", return_value=None):
            result = RUNNER.run_test(["test"], {}, output)
        self.assertEqual((result["outcome"], result["exit"]), ("timed-out", 124))
        self.assertLessEqual(process.communicate.call_args_list[1].kwargs["timeout"], RUNNER.CAPS["rust"])
        process.kill.assert_called_once()

    def test_native_deadline_still_stops_and_reaps_the_child(self):
        process = Mock()
        process.returncode = -9
        process.poll.return_value = None
        with TemporaryFile() as output, patch.object(RUNNER.sys, "platform", "linux"), \
                patch.object(RUNNER.subprocess, "Popen", return_value=process), \
                patch.object(RUNNER.resource, "getrusage", side_effect=[usage(), usage()]), \
                patch.object(RUNNER.time, "monotonic", side_effect=[0, RUNNER.CAPS["rust"], RUNNER.CAPS["rust"]]):
            result = RUNNER.run_test(["test"], {}, output)
        self.assertEqual(result["outcome"], "timed-out")
        self.assertEqual(result["exit"], 124)
        process.kill.assert_called_once()
        process.communicate.assert_called_once_with()

    def test_linux_residency_uses_kernel_kib_units(self):
        with patch.object(RUNNER.sys, "platform", "linux"), \
                patch.object(RUNNER.Path, "read_text", return_value="Name:\ttest\nVmRSS:\t7 kB\n"):
            self.assertEqual(RUNNER.resident_bytes(123, 300), 7 * 1024)

    def test_macos_ps_uses_kib_units(self):
        observed = subprocess.CompletedProcess([], 0, stdout=" 7\n", stderr="")
        with patch.object(RUNNER.sys, "platform", "darwin"), \
                patch.object(RUNNER.subprocess, "run", return_value=observed) as run:
            self.assertEqual(RUNNER.resident_bytes(123, 300), 7 * 1024)
            self.assertEqual(run.call_args.kwargs["timeout"], 300)

    def test_unsupported_platform_is_not_treated_as_measured(self):
        with patch.object(RUNNER.sys, "platform", "unsupported"), self.assertRaises(ValueError):
            RUNNER.run_test(["test"], {}, None)


if __name__ == "__main__":
    unittest.main()
