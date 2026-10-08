"""Fail-closed cases for the artifact digest and reuse checker."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_artifacts.py"
LEAN = "formal/nightstream-fprime/artifacts"
CRATE = "crates/nightstream/artifacts"


class CheckArtifacts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        script = self.root / "formal/nightstream-fprime/scripts/check_artifacts.py"
        script.parent.mkdir(parents=True)
        shutil.copy(SCRIPT, script)
        for directory in (LEAN, CRATE, "crates/nightstream-fprime/artifacts"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        self.write(f"{LEAN}/a.json", '{"v":1}')
        self.write(f"{CRATE}/shared.json", '{"s":1}')
        os.symlink("../../../formal/nightstream-fprime/artifacts/a.json", self.root / CRATE / "a.json")
        self.assert_passes("--write")

    def write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            path.unlink()
        path.write_text(text)

    def run_checker(self, *arguments):
        subprocess.run(["git", "add", "-A"], cwd=self.root, check=True)
        script = self.root / "formal/nightstream-fprime/scripts/check_artifacts.py"
        return subprocess.run([sys.executable, "-B", str(script), *arguments],
                              cwd=self.root, capture_output=True, text=True)

    def assert_passes(self, *arguments):
        result = self.run_checker(*arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("2 digests match; 1 links resolve", result.stdout)

    def assert_fails(self, message):
        result = self.run_checker()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn(message, result.stderr)

    def test_write_lists_only_regular_files(self):
        self.assertEqual((self.root / CRATE / "SHA256SUMS").read_text().count("\n"), 1)
        self.assert_passes()

    def test_changed_artifact_fails(self):
        self.write(f"{LEAN}/a.json", '{"v":2}')
        self.assert_fails(f"{LEAN}/a.json: digest")

    def test_unlisted_artifact_fails(self):
        self.write(f"{LEAN}/b.json", '{"b":1}')
        self.assert_fails(f"{LEAN}/b.json: not listed")

    def test_listed_symbolic_link_fails(self):
        manifest = self.root / CRATE / "SHA256SUMS"
        manifest.write_text(manifest.read_text() + f"{'0' * 64}  a.json\n")
        self.assert_fails(f"{CRATE}/a.json: listed in SHA256SUMS but is a symbolic link")

    def test_link_replaced_by_changed_file_fails(self):
        self.write(f"{CRATE}/a.json", '{"v":0}')
        self.assert_fails(f"{CRATE}/a.json: must be a symbolic link")

    def test_renamed_regular_copy_fails(self):
        self.write("crates/other/tests/fixtures/renamed.json", '{"v":1}')
        self.assert_fails("crates/other/tests/fixtures/renamed.json: must be a symbolic link")


if __name__ == "__main__":
    unittest.main()
