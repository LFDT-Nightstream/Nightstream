"""Owner-authorized deadline removal preserves the shared command guard."""
from contextlib import nullcontext
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from scripts.lean_graph import guard


class GuardDeadlineTests(unittest.TestCase):
    def test_no_timeout_keeps_lock_and_propagates_to_nested_lean(self):
        for kind, command in [
                ("lean", ["bash", "scripts/validate.sh", "build"]),
                ("python", ["python3", "check.py"]),
                ("rust", ["cargo", "test", "--release"])]:
            with self.subTest(kind=kind):
                child = MagicMock(pid=123)
                child.wait.return_value = 0
                with patch.object(guard, "build_lock", return_value=nullcontext()) as lock, \
                        patch.object(guard, "check_build_processes") as processes, \
                        patch.object(guard.subprocess, "Popen", return_value=child) as launch, \
                        patch.object(guard.os, "killpg"):
                    result = guard.run(command, kind, Path.cwd(), no_timeout=True)
                lock.assert_called_once()
                processes.assert_called_once()
                self.assertEqual(child.wait.call_args_list[0].kwargs, {"timeout": None})
                self.assertEqual(launch.call_args.kwargs["env"]["LEAN_TIMEOUT_SECONDS"], "0")
                self.assertIsNone(result["cap_seconds"])
                self.assertEqual(result["outcome"], "passed")


if __name__ == "__main__":
    unittest.main()
