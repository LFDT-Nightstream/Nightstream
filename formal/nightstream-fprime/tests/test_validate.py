"""Check the validation wrapper without invoking Lean or its build cache."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate.sh"


class ValidateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.lake = Path(self.directory.name) / "lake"
        self.lake.write_text(
            '#!/bin/sh\n'
            'printf "requested:%s\\n" "$@"\n'
            'for target do\n'
            '  case "$target" in\n'
            '    BrokenAudit) exit 7 ;;\n'
            '    SlowAudit) sleep 30 ;;\n'
            '  esac\n'
            'done\n'
        )
        self.lake.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": f"{self.directory.name}{os.pathsep}{os.environ['PATH']}",
            "LEAN_TIMEOUT_SECONDS": "1500",
        }

    def run_validation(self, *arguments, cap="1500"):
        return subprocess.run(
            ["bash", str(SCRIPT), *arguments],
            env={**self.env, "LEAN_TIMEOUT_SECONDS": cap},
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_build_forwards_all_requested_targets(self):
        result = self.run_validation("build", "Production", "FirstAudit", "SecondAudit")
        self.assertEqual(result.returncode, 0, result.stderr)
        requested = [line for line in result.stdout.splitlines() if line.startswith("requested:")]
        self.assertEqual(requested, [
            "requested:build", "requested:Production", "requested:FirstAudit", "requested:SecondAudit"
        ])

    def test_failure_in_later_target_fails_validation(self):
        result = self.run_validation("build", "Production", "BrokenAudit")
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertIn("exit=7", result.stdout)

    def test_default_build_target(self):
        result = self.run_validation("build")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("requested:NightstreamFPrime\n", result.stdout)

    def test_rejects_disabled_or_excessive_cap(self):
        for cap in ("0", "1501"):
            with self.subTest(cap=cap):
                result = self.run_validation("build", cap=cap)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("requested:", result.stdout)

    def test_timeout_fails_validation(self):
        result = self.run_validation("build", "SlowAudit", cap="1")
        self.assertEqual(result.returncode, 137, result.stdout + result.stderr)
        self.assertIn("TIMEOUT is a failed gate", result.stderr)


if __name__ == "__main__":
    unittest.main()
