"""Worktree isolation and owner-authorized deadline removal retain the guard."""
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts.lean_graph import guard


class GuardWorktreeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def checkout_guard(self, name):
        path = self.root / name / "scripts/lean_graph/guard.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(guard.__file__).read_bytes())
        spec = importlib.util.spec_from_file_location("scripts.lean_graph.checkout_guard", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.addCleanup(module._LOCK_PATH.unlink, missing_ok=True)
        return module

    def test_same_worktree_clients_share_lock_across_stores(self):
        first = self.checkout_guard("first")
        second = self.checkout_guard("first")
        with first.build_lock(self.root / "store-a"):
            with self.assertRaisesRegex(guard.EvidenceError, "this worktree"):
                with second.build_lock(self.root / "store-b"):
                    self.fail("same-worktree clients acquired the lock together")

    def test_different_worktrees_have_independent_locks(self):
        first = self.checkout_guard("first")
        second = self.checkout_guard("second")
        with first.build_lock(), second.build_lock():
            self.assertNotEqual(first._LOCK_PATH, second._LOCK_PATH)

    def test_symlink_checkout_uses_same_lock(self):
        first = self.checkout_guard("first")
        (self.root / "alias").symlink_to(self.root / "first", target_is_directory=True)
        alias = self.checkout_guard("alias")
        self.assertEqual(first._WORKTREE_ROOT, alias._WORKTREE_ROOT)
        self.assertEqual(first._LOCK_PATH, alias._LOCK_PATH)

    def test_same_worktree_unmanaged_build_is_rejected(self):
        process_list = MagicMock(stdout="12 lake\n13 /tools/rustc\n14 python3\n")
        with patch.object(guard.subprocess, "run", return_value=process_list), \
                patch.object(guard, "process_cwd", side_effect=[
                    guard._WORKTREE_ROOT / "formal/nightstream-fprime", self.root / "other"]):
            with self.assertRaisesRegex(guard.EvidenceError, "this worktree: 12 lake$"):
                guard.check_build_processes()

    def test_other_worktree_and_exited_builds_do_not_block(self):
        process_list = MagicMock(stdout="12 lean\n13 cargo\n14 rustc\n")
        with patch.object(guard.subprocess, "run", return_value=process_list), \
                patch.object(guard, "process_cwd", side_effect=[
                    self.root / "other", Path(str(guard._WORKTREE_ROOT) + "-other"), None]):
            guard.check_build_processes()

    @unittest.skipUnless(sys.platform == "linux", "Linux process cwd")
    def test_linux_reads_actual_process_working_directory(self):
        self.assertEqual(guard.process_cwd(os.getpid()), Path.cwd().resolve())

    def test_macos_resolves_reported_process_working_directory(self):
        result = MagicMock(returncode=0, stdout=f"p12\nfcwd\nn{self.root}\n")
        with patch.object(guard.sys, "platform", "darwin"), \
                patch.object(guard.subprocess, "run", return_value=result) as query:
            self.assertEqual(guard.process_cwd(12), self.root)
        self.assertEqual(query.call_args.args[0], ["lsof", "-a", "-p", "12", "-d", "cwd", "-Fn"])

    def test_macos_does_not_ignore_unobservable_live_build(self):
        result = MagicMock(returncode=1, stdout="")
        with patch.object(guard.sys, "platform", "darwin"), \
                patch.object(guard.subprocess, "run", return_value=result), patch.object(guard.os, "kill"):
            with self.assertRaisesRegex(guard.EvidenceError, "cannot determine the worktree"):
                guard.process_cwd(12)


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
